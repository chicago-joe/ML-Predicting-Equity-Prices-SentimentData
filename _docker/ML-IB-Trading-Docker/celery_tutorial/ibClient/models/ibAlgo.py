# --------------------------------------------------------------------------------------------------
# ibClient.py
#
# created by joe.loss
# --------------------------------------------------------------------------------------------------
# Module Imports

from ib_insync import *
import os, sys
from random import randint
from celery_tutorial.ibClient.util import order_util, dt_util
import pandas as pd
import numpy as np
import logging

import nest_asyncio
nest_asyncio.apply()

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# --------------------------------------------------------------------------------------------------
# base model class

class BaseModel(object):
    def __init__(self, host = '127.0.0.1', port = 7497, client_id = 1):
        self.host = host
        self.port = port
        self.client_id = client_id

        self.__ib = None
        self.pnl = None  # stores IB PnL object
        self.positions = { }  # stores IB Position object by symbol

        self.symbol_map = { }  # maps contract to symbol
        self.symbols, self.contracts = [], []
        self.pending_order_ids = set()

    def init_model(self, ticker_tk):
        """
        Initialize the model given inputs before running.
        Stores the input symbols and contracts that will be used for reading positions.

        :param ticker_tk: list of a tuple of symbol and contract, Example:
            [('EURUSD', Forex('EURUSD'), ]
        """

        ident = ticker_tk[0]
        contract = ticker_tk[1]
        self.symbol_map = { str(contract):ident }
        self.contracts = [contract]
        self.symbols = list(self.symbol_map.values())

    def connect_to_ib(self):
        # self.ib.reqCurrentTime()
        id = randint(0,1000)
        self.client_id = id

        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, self.client_id,timeout = 300)
            self.ib.waitOnUpdate(timeout=0.1)


    def on_pnl(self, pnl):
        """ Simply store a copy of the latest PnL whenever where are changes """
        self.pnl = pnl

    def request_position_updates(self):
        self.ib.reqPositions()
        self.ib.positionEvent += self.on_position

    # def onOrderStatus(self,trade):
    # 	trdStatus = trade.OrderStatus.status
    # 	# self.ib.reqPositions()
    # 	ib.orderStatusEvent += request_order_updates
    # 	self.ib.positionEvent += self.on_position

    def request_positions(self):
        return self.ib.positions()

    def request_portfolio(self):
        return self.ib.portfolio()

    def account_value(self):
        dfAccnt = pd.DataFrame(self.ib.accountValues()).groupby(['tag', 'currency'])['value'].sum().reset_index()
        dfAccnt['currency'] = np.where(dfAccnt['currency'] == '', 'USD', dfAccnt['currency'])
        dfAccnt = dfAccnt.loc[dfAccnt.currency == 'USD'].set_index('tag')['value']
        return dfAccnt

    def cash_balance(self):  # cash balance
        dfAccnt = self.account_value()
        portCash = dfAccnt.loc[dfAccnt.index == 'AvailableFunds']
        return float(portCash[0])

    def portfolio_net(self):
        dfAccnt = self.account_value()
        portNet = dfAccnt.loc[dfAccnt.index == 'NetLiquidation']
        return float(portNet[0])

    def on_position(self, position):
        """ Simply store a copy of the latest Position object for the provided contract """
        symbol = self.get_symbol(position.contract)
        if symbol not in self.symbols:
            logging.warning('Symbol not found for position: {}'.format(position))
            return

        self.positions[symbol] = position

    def request_all_contracts_data(self, fn_on_tick):
        for contract in self.contracts:
            self.ib.reqMarketDataType(1)
            self.ib.reqMktData(contract, '', False, False)
        self.ib.pendingTickersEvent += fn_on_tick

    def request_order_updates(self, fn_on_order):
        self.ib.orderStatusEvent += fn_on_order

    def request_exec_updates(self, fn_on_exec):
        self.ib.execDetailsEvent += fn_on_exec

    # --------------------------------------------------------------------------------------------------
    # place market order using SNAPMKT

    def place_market_order(self, contract, qty, fn_on_filled = None):

        if not fn_on_filled:
            fn_on_filled = self.fun_on_filled

        qty = float(qty)

        order = MarketOrder(
                order_util.get_order_action(qty),
                abs(qty),
                tif = 'DAY',
                outsideRth = True)
        order.orderType = 'SNAPMKT'

        trade = self.ib.placeOrder(contract, order)
        self.pending_order_ids.add(trade.order.orderId)
        trade.filledEvent += fn_on_filled
        return trade

    def get_price(self, instrument):
        contract1 = self.get_contract(instrument.symbol)
        self.ib.reqMarketDataType(1)
        data = self.ib.reqTickers(contract1)[0]
        return data.midpoint()

    def fun_on_filled(self, trade):
        logging.debug('Function Called on being filled')
        logging.info('Order filled: {}'.format(trade))

        self.pending_order_ids.remove(trade.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        # Update flag when all pending orders are filled
        if not self.pending_order_ids:
            self.is_orders_pending = False

    def get_contract(self, symbol):
        # return contract.Forex(symbol)
        return contract.Stock(symbol, 'SMART', 'USD')

    def get_symbol(self, contract):
        """
        Finds the symbol given the contract.

        :param contract: The Contract object
        :return: the symbol given for the specific contract
        """
        symbol = self.symbol_map.get(str(contract), None)
        if symbol:
            return symbol

        symbol = ''
        if type(contract) is Forex:
            symbol = contract.localSymbol.replace('.', '')
        elif type(contract) is Stock:
            symbol = contract.symbol

        return symbol if symbol in self.symbols else ''

    @property
    def ib(self):
        if not self.__ib:
            self.__ib = IB()
        return self.__ib


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# ib paper trading class

class HftModel1(BaseModel):

    def __init__(self, *args, **kwargs):
        super(HftModel1, self).__init__(*args, **kwargs)

        self.df_hist = None  # stores mid prices in a pandas DataFrame
        self.pending_order_ids = set()
        self.is_orders_pending = False
        self.exit = False
        self.position = 0
        self.prevPosition = 0
        self.is_buy_signal, self.is_sell_signal, self.is_flat_signal, self.is_partial_buy_signal = False, False, False, False

    def run(self, ticker_tk = None, position = 0, prevPosition = 0):
        """ Entry point """
        logging.info('Time started: {time}'.format(time = str(pd.to_datetime('now')), ))

        # Initialize model based on inputs
        self.init_model(ticker_tk)
        self.position = position
        self.prevPosition = prevPosition
        self.df_hist = pd.DataFrame(columns = self.symbols)

        # Establish connection to IB
        self.connect_to_ib()

        # self.request_pnl_updates()
        # self.request_position_updates()
        self.request_all_contracts_data(self.on_tick)
        self.request_order_updates(self.onOrderStatus)
        self.request_exec_updates(self.onExecDetails)

        # Recalculate and/or logging.info account updates at intervals
        for i in range (0,1):
            self.ib.sleep(i)
            self.perform_trade_logic()
        # while self.is_orders_pending:
        #     self.ib.sleep(1)

    def print_account(self):
        pd.set_option('display.max_rows', 120)
        logging.info('Account Summary:\n{}'.format(self.account_value()))
        pd.set_option('display.max_rows', 30)

    def on_tick(self, ticker_ticks):
        """ When a tick data is received, store it and make calculations out of it """
        for tick in ticker_ticks:
            self.get_incoming_tick_data(tick)
        # self.perform_trade_logic()

    def get_incoming_tick_data(self, ticker_tk):
        """
        Stores the midpoint of incoming price data to a pandas DataFrame `df_hist`.
        :param ticker_tk: The incoming tick data as a ticker_tk object.
        """
        symbol = self.get_symbol(ticker_tk.contract)
        dt_obj = dt_util.convert_utc_datetime(ticker_tk.time)

        bid = ticker_tk.bid
        ask = ticker_tk.ask

        if not ticker_tk.last.is_integer():
            last = ticker_tk.close
            self.df_hist.loc[dt_obj, 'last'] = last
        elif ticker_tk.last > 0:
            last = ticker_tk.last
            self.df_hist.loc[dt_obj, 'last'] = last
        if bid < 0 or ask < 0:
            mid = ticker_tk.close
        else:
            mid = (bid + ask) / 2
        self.df_hist.loc[dt_obj, 'mid'] = mid


    def onOrderStatus(self, trade):
        orderStatus = trade.orderStatus.status
        return

    def onExecDetails(self, trade, fill):
        execStatus = fill.execution
        # print(trade.orderStatus.status)
        return

    def on_filled(self, trade):
        logging.info('Order filled: {}'.format(trade))
        self.pending_order_ids.remove(trade.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        # Update flag when all pending orders are filled
        if not self.pending_order_ids:
            self.is_orders_pending = False

    def on_spread_leg_filled(self, trade):
        logging.info('Spread Leg 1 closed: {}'.format(trade))
        self.pending_order_ids.remove(trade.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        # Update flag when all pending orders are filled
        if not self.pending_order_ids:
            self.is_orders_pending = False

    def calculate_signals(self):
        self.is_flat_signal = self.position == 0
        self.is_buy_signal = self.position == 1
        self.is_partial_buy_signal = self.position == 0.75
        self.is_sell_signal = self.position == -1

    # --------------------------------------------------------------------------------------------------
    # calculate signal using position from smadb.tbllivepositionsignal

    def perform_trade_logic(self):

        self.calculate_signals()
        # for i in range(0,1):
        #     if self.is_orders_pending or self.check_and_enter_orders():
        #         return
        self.check_and_enter_orders()
        while self.is_orders_pending:
            self.ib.sleep(1)

        logging.info('\n\n----- TRADES COMPLETED. ENDING PROGRAM NOW. -----\n\n')
        self.ib.disconnect()
        sys.exit()


    def check_and_enter_orders(self):
        if self.is_position_flat and self.is_sell_signal:
            logging.info('*** OPENING SHORT POSITION ***')
            self.place_order_to_open(self.position)
            return True

        if self.is_position_flat and self.is_buy_signal:
            logging.info('*** OPENING LONG POSITION ***')
            self.place_order_to_open(self.position)
            return True

        if self.is_position_flat and self.is_partial_buy_signal:
            logging.info('*** OPENING PARTIAL LONG POSITION ***')
            self.place_order_to_open(self.position)
            return True

        if self.is_position_flat and self.is_flat_signal:
            logging.info('*** REMAINING FLAT, NO POSITION OPEN ***')
            return False

        if self.is_position_long and self.is_buy_signal:
            if self.position == self.prevPosition:
                logging.info('*** REMAINING LONG ***')
                return False
            else:
                logging.info('*** ADJUSTING LONG ALLOCATION: Current: {:,.0f}% Target: {:,.0f}% ***'.format(self.prevPosition * 100, self.position * 100))
                self.place_adjustment_order(self.position, self.prevPosition)
                return True

        if self.is_position_long and self.is_partial_buy_signal:
            if self.position == self.prevPosition:
                logging.info('*** REMAINING LONG ***')
                return False
            else:
                logging.info('*** ADJUSTING LONG ALLOCATION: Current: {:,.0f}% Target: {:,.0f}% ***'.format(self.prevPosition * 100, self.position * 100))
                self.place_adjustment_order(self.position, self.prevPosition)
                return True

        if self.is_position_long and self.is_sell_signal:
            logging.info('*** OPENING SHORT POSITION ***')
            self.place_spread_order(self.position)
            return True

        if self.is_position_long and self.is_flat_signal:
            logging.info('*** CLOSING LONG POSITION ***')
            self.place_order_to_close()
            return True

        if self.is_position_short and self.is_sell_signal:
            logging.info('*** REMAINING SHORT ***')
            return False

        if self.is_position_short and self.is_buy_signal:
            logging.info('*** OPENING LONG POSITION ***')
            self.place_spread_order(self.position)
            return True

        if self.is_position_short and self.is_partial_buy_signal:
            logging.info('*** OPENING PARTIAL LONG POSITION ***')
            self.place_spread_order(self.position)
            return True

        if self.is_position_short and self.is_flat_signal:
            self.place_order_to_close()
            return True
        return False

    def place_order_to_open(self, pos):

        cnOrder = self.contracts[0]
        dfAccnt = self.account_value()
        cash = dfAccnt.loc[dfAccnt.index == 'BuyingPower'].astype(float)
        last = self.df_hist['last'][0].astype(float)

        if pos < 0:
            qty = (cash / last) * abs(pos) * -1
        else:
            qty = (cash / last) * abs(pos) * 1

        qty = int(qty)

        tradeOpen = self.place_market_order(cnOrder, qty, self.on_filled)
        logging.info('Order placed: {}'.format(tradeOpen))

        self.is_orders_pending = True
        self.pending_order_ids.add(tradeOpen.order.orderId)

        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))
        return

    def place_order_to_close(self):

        cnOrder = self.contracts[0]
        currentPos = self.ib.positions()[0].position

        tradeClose = self.place_market_order(cnOrder, -currentPos, self.on_filled)
        logging.info('Order placed: {}'.format(tradeClose))

        self.is_orders_pending = True
        self.pending_order_ids.add(tradeClose.order.orderId)

        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))
        return

    def place_spread_order(self, pos):

        logging.info('Placing spread orders...')
        cnOrder = self.contracts[0]

        # close current position
        currentPos = self.ib.positions()[0].position
        tradeClose = self.place_market_order(cnOrder, -currentPos, self.on_filled)
        logging.info('Order placed: {}'.format(tradeClose))

        self.is_orders_pending = True
        self.pending_order_ids.add(tradeClose.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        # todo:
        while not tradeClose.isDone():
            self.ib.waitOnUpdate()

        dfAccnt = self.account_value()
        cash = dfAccnt.loc[dfAccnt.index == 'BuyingPower'].astype(float)
        last = self.df_hist['last'][0].astype(float)

        # open new position
        if pos < 0:
            qty = (cash / last) * abs(pos) * -1
        else:
            qty = (cash / last) * abs(pos) * 1

        qty = int(qty)

        tradeOpen = self.place_market_order(cnOrder, qty, self.on_filled)
        logging.info('Order placed: {}'.format(tradeOpen))

        self.is_orders_pending = True
        self.pending_order_ids.add(tradeOpen.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        return

    # places an adjustment order to go from 100% to 75% invested and vice-versa
    def place_adjustment_order(self, pos, posPr):

        cnOrder = self.contracts[0]
        port = self.ib.portfolio()[0]

        currentAlloc = posPr
        targetAlloc = pos
        currentMV = port.marketValue
        currentShares = port.position

        if targetAlloc < currentAlloc:
            dAlloc = currentAlloc - targetAlloc
            target = (1 - dAlloc) * currentMV
        else:
            dAlloc = targetAlloc + (targetAlloc - currentAlloc)
            target = dAlloc * currentMV

        last = self.df_hist['last'][0].astype(float)

        if target < currentMV:
            targetShares = target / last
            trdShares = targetShares - currentShares
        else:
            dfAccnt = self.account_value()
            cash = dfAccnt.loc[dfAccnt.index == 'BuyingPower'].astype(float)
            last = self.df_hist['last'][0].astype(float)

            # open new position
            if pos < 0:
                trdShares = (cash / last) * abs(pos) * -1
            else:
                trdShares = (cash / last) * abs(pos) * 1

        trdShares = int(trdShares)

        trade = self.place_market_order(cnOrder, trdShares, self.on_filled)
        logging.info('Order placed: {}'.format(trade))

        self.is_orders_pending = True
        self.pending_order_ids.add(trade.order.orderId)
        logging.info('Order IDs pending execution: {}'.format(self.pending_order_ids))

        return

    # --------------------------------------------------------------------------------------------------
    # properties

    @property
    def is_position_flat(self):
        if not self.ib.positions():
            return True

    @property
    def is_position_short(self):
        if self.ib.positions()[0].position < 0:
            return True

    @property
    def is_position_long(self):
        if self.ib.positions()[0].position > 0:
            return True
