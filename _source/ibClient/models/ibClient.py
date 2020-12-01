"""
A base model containing common IB functions.

For other models to extend and use.
"""

from ib_insync import *
from util import order_util, dt_util
import pandas as pd


# --------------------------------------------------------------------------------------------------
# base model class

class BaseModel(object):
	def __init__(self, host='127.0.0.1', port=7496, client_id=1):
		self.host = host
		self.port = port
		self.client_id = client_id

		self.__ib = None
		self.pnl = None  # stores IB PnL object
		self.positions = {}  # stores IB Position object by symbol

		self.symbol_map = {}  # maps contract to symbol
		self.symbols, self.contracts = [], []
		self.pending_order_ids = set()

	def init_model(self, to_trade):
		"""
		Initialize the model given inputs before running.
		Stores the input symbols and contracts that will be used for reading positions.

		:param to_trade: list of a tuple of symbol and contract, Example:
			[('EURUSD', Forex('EURUSD'), ]
		"""
		self.symbol_map = {str(contract): ident for (ident, contract) in to_trade}
		self.contracts = [contract for (_, contract) in to_trade]
		self.symbols = list(self.symbol_map.values())

	def connect_to_ib(self):
		self.ib.connect(self.host, self.port, clientId=self.client_id)

	def request_pnl_updates(self):
		account = self.ib.managedAccounts()[0]
		self.ib.reqPnL(account)
		self.ib.pnlEvent += self.on_pnl

	def on_pnl(self, pnl):
		""" Simply store a copy of the latest PnL whenever where are changes """
		self.pnl = pnl

	def request_position_updates(self):
		self.ib.reqPositions()
		self.ib.positionEvent += self.on_position

	def request_positions(self):
		return self.ib.positions()

	def request_portfolio(self):
		return self.ib.portfolio()

	def account_value(self):
		dfAccnt = pd.DataFrame(self.ib.accountValues()).groupby(['account', 'tag'])['value'].sum()
		return dfAccnt

	def cash_balance(self):# cash balance
		dfAccnt = self.account_value()
		portCash = dfAccnt.loc[dfAccnt.index.get_level_values(1) == 'AvailableFunds']
		return float(portCash[0])

	def portfolio_net(self):
		dfAccnt = self.account_value()
		portNet = dfAccnt.loc[dfAccnt.index.get_level_values(1) == 'NetLiquidation']
		return float(portNet[0])

	def on_position(self, position):
		""" Simply store a copy of the latest Position object for the provided contract """
		symbol = self.get_symbol(position.contract)
		if symbol not in self.symbols:
			print('[warn]symbol not found for position:', position)
			return

		self.positions[symbol] = position

	def request_all_contracts_data(self, fn_on_tick):
		for contract in self.contracts:
			self.ib.reqMarketDataType(1)
			self.ib.reqMktData(contract,)
			# self.ib.reqMktData(contract,'',False,False)

		self.ib.pendingTickersEvent += fn_on_tick

	# def reqMarketDataType(self, marketDataType):
	# 	for contract in self.contracts:
	# 		self.ib.reqMktDataType(contract,)

		# self.ib.send(59,1,3)
		# self.send(59,1,3)

	def place_market_order(self, contract, qty, fn_on_filled=None):
		if not fn_on_filled:
			fn_on_filled = self.fun_on_filled
		order = MarketOrder(order_util.get_order_action(qty), abs(qty))
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
		print('Function Called on being filled')
		print('Order filled:', trade)
		self.pending_order_ids.remove(trade.order.orderId)
		print('Order IDs pending execution:', self.pending_order_ids)

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
		super().__init__(*args, **kwargs)

		self.df_hist = None  # stores mid prices in a pandas DataFrame
		self.pending_order_ids = set()
		self.is_orders_pending = False
		self.trade_qty = 0


	def run(self, to_trade=[], trade_qty=0):
		""" Entry point """
		print('[{time}]started'.format(
			time=str(pd.to_datetime('now')),
		))

		# Initialize model based on inputs
		self.init_model(to_trade)
		self.trade_qty = trade_qty
		self.df_hist = pd.DataFrame(columns=self.symbols)

		# Establish connection to IB
		self.connect_to_ib()
		self.request_pnl_updates()
		self.request_position_updates()
		self.request_all_contracts_data(self.on_tick)

		# Recalculate and/or print account updates at intervals
		while self.ib.waitOnUpdate():
			self.ib.sleep(1)

			if not self.is_position_flat:
				self.print_account()


	def on_tick(self, tickers):
		""" When a tick data is received, store it and make calculations out of it """
		for ticker in tickers:
			self.get_incoming_tick_data(ticker)

		self.perform_trade_logic()


	def perform_trade_logic(self):

		if self.is_orders_pending or self.check_and_enter_orders():
			return  # Do nothing while waiting for orders to be filled

		if self.is_position_flat:
			print('--- no position -----')
			# self.print_strategy_params()


	def print_account(self):

		print('---- insert accnt info here ----')

		# todo: print current positon / ticker price info
		# position=100.0, marketPrice=366.14001465, marketValue=36614.0, averageCost=303.491603, unrealizedPNL=6264.84, realizedPNL=0.0
		# print(self.request_portfolio()[0][1:-1])



	# def print_strategy_params(self):
	# 	print('[{time}][strategy params]beta:{beta:.2f} volatility:{vr:.2f}|rpnl={rpnl:.2f}'.format(
	# 		time=str(pd.to_datetime('now')),
	# 		rpnl=self.pnl.realizedPnL,
	# 	))

	def check_and_enter_orders(self):
		if self.is_position_flat and self.is_sell_signal:
			print('*** OPENING SHORT POSITION ***')
			self.place_spread_order(-self.trade_qty)
			return True

		if self.is_position_flat and self.is_buy_signal:
			print('*** OPENING LONG POSITION ***')
			self.place_spread_order(self.trade_qty)
			return True

		if self.is_position_short and self.is_buy_signal:
			print('*** CLOSING SHORT POSITION ***')
			self.place_spread_order(self.trade_qty)
			return True

		if self.is_position_long and self.is_sell_signal:
			print('*** CLOSING LONG POSITION ***')
			self.place_spread_order(-self.trade_qty)
			return True

		return False


	def place_spread_order(self, qty):

		print('Placing spread orders...')
		[contract_a, contract_b] = self.contracts

		# self.place_market_order(self.contracts[0],-100)
		# Trade(contract=Stock(symbol='SPY', exchange='SMART', currency='USD'), order=MarketOrder(orderId=13, clientId=2, action='SELL', totalQuantity=100), orderStatus=OrderStatus(orderId=13, status='PendingSubmit', filled=0, remaining=0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', mktCapPrice=0.0), fills=[], log=[TradeLogEntry(time=datetime.datetime(2020, 12, 1, 12, 6, 57, 927050, tzinfo=datetime.timezone.utc), status='PendingSubmit', message='')])

		trade_a = self.place_market_order(contract_a, qty, self.on_filled)
		print('Order placed:', trade_a)


		self.is_orders_pending = True
		self.pending_order_ids.add(trade_a.order.orderId)
		print('Order IDs pending execution:', self.pending_order_ids)


	def on_filled(self, trade):
		print('Order filled:', trade)
		self.pending_order_ids.remove(trade.order.orderId)
		print('Order IDs pending execution:', self.pending_order_ids)

		# Update flag when all pending orders are filled
		if not self.pending_order_ids:
			self.is_orders_pending = False


	def get_incoming_tick_data(self, ticker):
		"""
		Stores the midpoint of incoming price data to a pandas DataFrame `df_hist`.
		:param ticker: The incoming tick data as a Ticker object.
		"""
		symbol = self.get_symbol(ticker.contract)

		dt_obj = dt_util.convert_utc_datetime(ticker.time)
		bid = ticker.bid
		ask = ticker.ask
		mid = (bid + ask) / 2
		self.df_hist.loc[dt_obj, symbol] = mid


	@property
	def is_position_flat(self):
		position_obj = self.positions.get(self.symbols[0])
		if not position_obj:
			return True
		return position_obj.position == 0

	@property
	def is_position_short(self):
		position_obj = self.positions.get(self.symbols[0])
		return position_obj and position_obj.position < 0

	@property
	def is_position_long(self):
		position_obj = self.positions.get(self.symbols[0])
		return position_obj and position_obj.position > 0










# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#

		# [symbol_a, symbol_b] = self.symbols
		# position_a, position_b = self.positions.get(symbol_a), self.positions.get(symbol_b)
		#
		# print('[{time}][account]{symbol_a} pos={pos_a} avgPrice={avg_price_a}|'
		# 	  '{symbol_b} pos={pos_b}|rpnl={rpnl:.2f} upnl={upnl:.2f}|beta:{beta:.2f} volatility:{vr:.2f}'.format(
		# 	time=str(pd.to_datetime('now')),
		# 	symbol_a=symbol_a,
		# 	pos_a=position_a.position if position_a else 0,
		# 	avg_price_a=position_a.avgCost if position_a else 0,
		# 	symbol_b=symbol_b,
		# 	pos_b=position_b.position if position_b else 0,
		# 	avg_price_b=position_b.avgCost if position_b else 0,
		# 	rpnl=self.pnl.realizedPnL,
		# 	upnl=self.pnl.unrealizedPnL,
		# 	beta=self.beta,
		# 	vr=self.volatility_ratio,
		# ))



	# def recalculate_strategy_params(self):
	# 	""" Calculating beta and volatility ratio for our signal indicators """
	# 	[symbol_a, symbol_b] = self.symbols
	#
	# 	resampled = self.df_hist.resample('30s').ffill().dropna()
	# 	mean = resampled.mean()
	# 	self.beta = mean[symbol_a] / mean[symbol_b]
	#
	# 	stddevs = resampled.pct_change().dropna().std()
	# 	self.volatility_ratio = stddevs[symbol_a] / stddevs[symbol_b]



	# def trim_historical_data(self):
	# 	""" Ensure historical data don't grow beyond a certain size """
	# 	cutoff_time = dt.datetime.now(tz=dt_util.LOCAL_TIMEZONE) - self.moving_window_period
	# 	self.df_hist = self.df_hist[self.df_hist.index >= cutoff_time]

		# for bar in bars:
		# 	dt_obj = dt_util.convert_local_datetime(bar.date)
		# 	self.df_hist.loc[dt_obj, symbol] = bar.close
