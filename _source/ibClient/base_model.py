
"""
A base model containing common IB functions.

For other models to extend and use.
"""

from ib_insync import IB, Forex, Stock, MarketOrder, contract
from  util import order_util
import pandas as pd
# --------------------------------------------------------------------------------------------------
# base model class

class BaseModel(object):
	def __init__(self, host='127.0.0.1', port=4002, client_id=1):
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
			self.ib.reqMarketDataType(3)
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
		self.ib.reqMarketDataType(3)
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
