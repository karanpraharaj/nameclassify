#! /usr/bin/python

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, data_size, hidden_size, output_size):
		super(RNN,self).__init__()

		self.hidden_size = hidden_size
		input_size = data_size + hidden_size

		self.i2h = nn.Linear(input_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)

#		self.hidden_size = hidden_size
#		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		
		self.softmax = nn.LogSoftmax(dim = 1)
		
	def forward(self, data, hidden):
		combined = torch.cat((data,hidden), 1)
		#hidden = self.i2h(combined)
		#output = self.i2o(combined)

		hidden = self.i2h(combined)
		output = self.h2o(hidden)
		output = self.softmax(output)
		return hidden, output

	def initHidden(self):
		return Variable(torch.zeros(1,self.hidden_size))
	

	
		
