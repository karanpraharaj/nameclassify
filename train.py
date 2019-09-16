
#! /usr/bin/python 
from __future__ import print_function
from preprocess import *
from model import *
import random
import torch, time, math
import torch.optim as optim

n_epochs = 100000
n_hidden = 128
print_every = 5000
plot_every = 10000
learning_rate = 0.005

def categoryFromOutput(output):
	a, b = output.topk(1)
	b = b[0].item()
	return all_categories[b], b

def randomChoice(l):
	return l[random.randint(0, len(l)-1)]

def randomExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
	line_tensor = Variable(linetotensor(line))
	return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
	hidden = rnn.initHidden()
	optimizer.zero_grad()
	
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)
	
	loss = criterion(output, category_tensor)
	loss.backward()
	
	optimizer.step()

	return output, loss.data[0]

# Keeping track of losses for plotting
current_loss = 0
all_losses = []

def timesince(since):
	now = time.time()
	s = now - since
	m = math.floor(s/60)
	s = s - 60*m
	return "%dm  %ds" %(m,s)

start = time.time()

for epoch in range(n_epochs):
	category, line, category_tensor, line_tensor = randomExample()
	output, loss = train(category_tensor, line_tensor)
	current_loss += loss
	
	if (epoch % print_every == 0) :
		guess, guess_i = categoryFromOutput(output)
		if guess == category:
			correct = '✓'
		else:
			correct = '✗ (%s)' % category
		print("%d %d%% (%s) %.4f %s / %s %s") % (epoch, epoch/n_epochs, timesince(start), loss, line, guess, correct)

	if epoch % plot_every == 0 :
		all_losses.append(current_loss/plot_every)
		current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

