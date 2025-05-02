import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from functional import LinearIF, AvgPool2d_IF, Conv2dIF, Conv2dIF_v2,activate,ConvTranspose2dIF,ConvTranspose2dIF_v2#,OperationFCIF,OperationConv2dIF
import numpy as np
import copy
from settings import SETTINGS
# from operationNN import OperationFC,OperationConv2d

class ConvBN2d_v3(nn.Module):

	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
				 padding=0, bias=True, weight_init=1.0, eps=1e-4, momentum=0.9,last_layer=False,pooling=1):
		super(ConvBN2d_v3, self).__init__()
		self.conv2dIF = Conv2dIF.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		if not last_layer:
			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
			# nn.init.kaiming_uniform_(self.conv2d.weight, mode='fan_in', nonlinearity='relu')

		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling

		self.laset_layer = last_layer
		# if last_layer:
		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
		nn.init.uniform_(self.conv2d.weight, -0.5, 0.5)
		nn.init.uniform_(self.conv2d.bias, -0.5, 0.5)

		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		# nn.init.zeros_(self.conv2d.bias)

	def forward(self, x, input_feature_st, input_features_sc):
		# T = input_feature_st.shape[1]
		if self.laset_layer:
			x = F.max_pool2d((self.conv2d(x)),self.pooling)#self.bn1d
			x_sc = F.max_pool2d(self.conv2d(input_features_sc),self.pooling)
			return x, input_feature_st, x_sc

		# weight update based on the surrogate conv2d layer
		output = F.relu(F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)),self.pooling))
		# output = torch.clamp(output_bn, min=0, max=T)
		x = F.relu(F.max_pool2d(self.bn2d(self.conv2d(x)),self.pooling))

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2)
		biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, \
															   weightNorm, self.device, \
															   biasNorm, self.stride, self.padding,self.pooling)

		return x*0.5+output*0.5, output_features_st, output_features_sc

class ConvTransposeBN2d_v3(nn.Module):

	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
				 padding=0, bias=True, weight_init=2.0, eps=1e-4, momentum=0.9,output_padding=0,last_layer=False, pooling=1):
		super(ConvTransposeBN2d_v3, self).__init__()
		self.convTranspose2dIF = ConvTranspose2dIF.apply

			# nn.init.normal_(self.bn2d.weight, 0, 2.0)
			# nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
			# nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
			# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
			# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		self.convTranspose2d = torch.nn.ConvTranspose2d(Cin, Cout, kernel_size, stride, padding, output_padding, bias=bias)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.output_padding = output_padding
		self.pooling = pooling
		self.laset_layer = last_layer
		if not last_layer:
			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
			# nn.init.kaiming_uniform_(self.convTranspose2d.weight, mode='fan_in', nonlinearity='relu')

		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
		# if last_layer:
		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
		nn.init.uniform_(self.convTranspose2d.weight, -0.5, 0.5)
		nn.init.uniform_(self.convTranspose2d.bias, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		# nn.init.zeros_(self.convTranspose2d.bias)

	def forward(self, x, input_feature_st, input_features_sc):
		# T = input_feature_st.shape[1]
		if self.laset_layer:
			x = F.max_pool2d((self.convTranspose2d(x)),self.pooling)#self.bn1d
			x_sc = F.max_pool2d(self.convTranspose2d(input_features_sc),self.pooling)
			return x, input_feature_st, x_sc

		# weight update based on the surrogate conv2d layer
		output = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(input_features_sc)),self.pooling))
		# output = torch.clamp(output_bn, min=0, max=T)
		x = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(x)),self.pooling))

		# extract the weight and bias from the surrogate conv layer
		convTranspose2d_weight = self.convTranspose2d.weight.detach().to(self.device)
		convTranspose2d_bias = self.convTranspose2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(convTranspose2d_weight.permute(0,2,3,1), ratio).permute(0,3,1,2)
		biasNorm = torch.mul(convTranspose2d_bias - bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.convTranspose2dIF(input_feature_st, output, \
															   weightNorm, self.device, \
															   biasNorm, self.stride, self.padding,self.output_padding,self.pooling)

		return x*0.5+output*0.5, output_features_st, output_features_sc







# class ConvTransposeBN2d_v3(nn.Module):
#
# 	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
# 				 padding=0, bias=True, weight_init=2.0, eps=1e-4, momentum=0.9,output_padding=0,last_layer=False, pooling=1):
# 		super(ConvTransposeBN2d_v3, self).__init__()
# 		self.convTranspose2dIF_v2 = ConvTranspose2dIF_v2.apply
#
# 			# nn.init.normal_(self.bn2d.weight, 0, 2.0)
# 			# nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
# 			# nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
# 			# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
# 			# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)
#
# 		self.convTranspose2d = torch.nn.ConvTranspose2d(Cin, Cout, kernel_size, stride, padding, output_padding, bias=bias)
# 		self.device = device
# 		self.stride = stride
# 		self.padding = padding
# 		self.output_padding = output_padding
# 		self.pooling = pooling
# 		self.laset_layer = last_layer
# 		if not last_layer:
# 			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
# 			# nn.init.kaiming_uniform_(self.convTranspose2d.weight, mode='fan_in', nonlinearity='relu')
#
# 		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
# 		# if last_layer:
# 		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
# 		nn.init.uniform_(self.convTranspose2d.weight, -0.5, 0.5)
# 		nn.init.uniform_(self.convTranspose2d.bias, -0.5, 0.5)
# 		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
# 		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)
#
# 		# nn.init.zeros_(self.convTranspose2d.bias)
#
# 	def forward(self, x, input_feature_st, input_features_sc):
# 		# T = input_feature_st.shape[1]
# 		if self.laset_layer:
# 			x = F.max_pool2d((self.convTranspose2d(x)),self.pooling)#self.bn1d
# 			x_sc = F.max_pool2d(self.convTranspose2d(input_features_sc),self.pooling)
# 			return x, input_feature_st, x_sc
#
# 		# weight update based on the surrogate conv2d layer
# 		output = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(input_features_sc)),self.pooling))
# 		# output = torch.clamp(output_bn, min=0, max=T)
# 		x = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(x)),self.pooling))
#
# 		# extract the weight and bias from the surrogate conv layer
# 		convTranspose2d_weight = self.convTranspose2d.weight.detach().to(self.device)
# 		convTranspose2d_bias = self.convTranspose2d.bias.detach().to(self.device)
#
# 		bnGamma = self.bn2d.weight
# 		bnBeta = self.bn2d.bias
# 		bnMean = self.bn2d.running_mean
# 		bnVar = self.bn2d.running_var
#
# 		# re-parameterization by integrating the beta and gamma factors
# 		# into the 'Conv' layer weights
# 		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
# 		weightNorm = torch.mul(convTranspose2d_weight.permute(0,2,3,1), ratio).permute(0,3,1,2)
# 		biasNorm = torch.mul(convTranspose2d_bias - bnMean, ratio) + bnBeta
#
# 		# propagate the input spike train through the IF layer to get actual output
# 		# spike train
# 		output_features_st, output_features_sc = self.convTranspose2dIF_v2(input_feature_st, output, \
# 															   weightNorm, self.device, \
# 															   biasNorm, self.stride, self.padding,self.output_padding,self.pooling,self.training)
#
# 		return x, output_features_st, output_features_sc

# class ConvBN2d_v3(nn.Module):
#
# 	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
# 				 padding=0, bias=True, weight_init=1.0, eps=1e-4, momentum=0.9,last_layer=False,pooling=1):
# 		super(ConvBN2d_v3, self).__init__()
# 		self.conv2dIF_v2 = Conv2dIF_v2.apply
# 		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
# 		if not last_layer:
# 			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
# 			nn.init.kaiming_uniform_(self.conv2d.weight, mode='fan_in', nonlinearity='relu')
#
# 		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
# 		self.device = device
# 		self.stride = stride
# 		self.padding = padding
# 		self.pooling = pooling
#
# 		self.laset_layer = last_layer
# 		# if last_layer:
# 		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
# 		nn.init.uniform_(self.conv2d.weight, -0.5, 0.5)
# 		nn.init.uniform_(self.conv2d.bias, -0.5, 0.5)
#
# 		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
# 		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)
#
# 		# nn.init.zeros_(self.conv2d.bias)
#
# 	def forward(self, x, input_feature_st, input_features_sc):
# 		# T = input_feature_st.shape[1]
# 		if self.laset_layer:
# 			x = F.max_pool2d((self.conv2d(x)),self.pooling)#self.bn1d
# 			x_sc = F.max_pool2d(self.conv2d(input_features_sc),self.pooling)
# 			return x, input_feature_st, x_sc
#
# 		# weight update based on the surrogate conv2d layer
# 		output = F.relu(F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)),self.pooling))
# 		# output = torch.clamp(output_bn, min=0, max=T)
# 		x = F.relu(F.max_pool2d(self.bn2d(self.conv2d(x)),self.pooling))
#
# 		# extract the weight and bias from the surrogate conv layer
# 		conv2d_weight = self.conv2d.weight.detach().to(self.device)
# 		conv2d_bias = self.conv2d.bias.detach().to(self.device)
#
# 		bnGamma = self.bn2d.weight
# 		bnBeta = self.bn2d.bias
# 		bnMean = self.bn2d.running_mean
# 		bnVar = self.bn2d.running_var
#
# 		# re-parameterization by integrating the beta and gamma factors
# 		# into the 'Conv' layer weights
# 		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
# 		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2)
# 		biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta
#
# 		# propagate the input spike train through the IF layer to get actual output
# 		# spike train
# 		output_features_st, output_features_sc = self.conv2dIF_v2(input_feature_st, output, \
# 															   weightNorm, self.device, \
# 															   biasNorm, self.stride, self.padding,self.pooling,self.training)
#
# 		return x, output_features_st, output_features_sc










################ using in SpikeVAE ###############################
class ConvTransposeBN2d_v2(nn.Module):

	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
				 padding=0, bias=True, weight_init=2.0, eps=1e-4, momentum=0.9,output_padding=0,last_layer=False, pooling=1):
		super(ConvTransposeBN2d_v2, self).__init__()
		self.convTranspose2dIF = ConvTranspose2dIF.apply

			# nn.init.normal_(self.bn2d.weight, 0, 2.0)
			# nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
			# nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
			# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
			# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		self.convTranspose2d = torch.nn.ConvTranspose2d(Cin, Cout, kernel_size, stride, padding, output_padding, bias=bias)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.output_padding = output_padding
		self.pooling = pooling
		self.laset_layer = last_layer
		if not last_layer:
			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
			# nn.init.kaiming_uniform_(self.convTranspose2d.weight, mode='fan_in', nonlinearity='relu')

		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
		# if last_layer:
		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
		nn.init.uniform_(self.convTranspose2d.weight, -0.5, 0.5)
		nn.init.uniform_(self.convTranspose2d.bias, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		# nn.init.zeros_(self.convTranspose2d.bias)

	def forward(self, x, input_feature_st, input_features_sc):
		# T = input_feature_st.shape[1]
		if self.laset_layer:
			x = F.max_pool2d((self.convTranspose2d(x)),self.pooling)#self.bn1d
			x_sc = F.max_pool2d(self.convTranspose2d(input_features_sc),self.pooling)
			return x, input_feature_st, x_sc

		# weight update based on the surrogate conv2d layer
		output = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(input_features_sc)),self.pooling))
		# output = torch.clamp(output_bn, min=0, max=T)
		x = F.relu(F.max_pool2d(self.bn2d(self.convTranspose2d(x)),self.pooling))

		# extract the weight and bias from the surrogate conv layer
		convTranspose2d_weight = self.convTranspose2d.weight.detach().to(self.device)
		convTranspose2d_bias = self.convTranspose2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(convTranspose2d_weight.permute(0,2,3,1), ratio).permute(0,3,1,2)
		biasNorm = torch.mul(convTranspose2d_bias - bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.convTranspose2dIF(input_feature_st, output, \
															   weightNorm, self.device, \
															   biasNorm, self.stride, self.padding,self.output_padding,self.pooling)

		return x, output_features_st, output_features_sc

class ConvBN2d_v2(nn.Module):

	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
				 padding=0, bias=True, weight_init=1.0, eps=1e-4, momentum=0.9,last_layer=False,pooling=1):
		super(ConvBN2d_v2, self).__init__()
		self.conv2dIF = Conv2dIF.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		if not last_layer:
			self.bn2d = torch.nn.BatchNorm2d(Cout, eps=eps, momentum=momentum)
			# nn.init.kaiming_uniform_(self.conv2d.weight, mode='fan_in', nonlinearity='relu')

		# nn.init.normal_(self.bn2d.weight, 0, weight_init)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling

		self.laset_layer = last_layer
		# if last_layer:
		#     nn.init.normal_(self.bn1d.weight, 0, 2.0)
		nn.init.uniform_(self.conv2d.weight, -0.5, 0.5)
		nn.init.uniform_(self.conv2d.bias, -0.5, 0.5)

		# nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
		# nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

		# nn.init.zeros_(self.conv2d.bias)

	def forward(self, x, input_feature_st, input_features_sc):
		# T = input_feature_st.shape[1]
		if self.laset_layer:
			x = F.max_pool2d((self.conv2d(x)),self.pooling)#self.bn1d
			x_sc = F.max_pool2d(self.conv2d(input_features_sc),self.pooling)
			return x, input_feature_st, x_sc

		# weight update based on the surrogate conv2d layer
		output = F.relu(F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)),self.pooling))
		# output = torch.clamp(output_bn, min=0, max=T)
		x = F.relu(F.max_pool2d(self.bn2d(self.conv2d(x)),self.pooling))

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2)
		biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, \
															   weightNorm, self.device, \
															   biasNorm, self.stride, self.padding,self.pooling)

		return x, output_features_st, output_features_sc

class ConvTransposeBN2d(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0, output_padding=0,bias=True, weight_init=2.0, pooling=1):

		super(ConvTransposeBN2d, self).__init__()
		self.convTranspose2dIF = ConvTranspose2dIF.apply
		self.convTranspose2d = torch.nn.ConvTranspose2d(Cin, Cout, kernel_size, stride, padding, output_padding, bias=bias)
		self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
		# self.bn2d = torch.nn.BatchNorm2d(Cout)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling
		self.output_padding = output_padding
		nn.init.normal_(self.bn2d.weight, 0, weight_init)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate conv2d layer
		output_bn = F.max_pool2d(self.bn2d(self.convTranspose2d(input_features_sc)), self.pooling)
		output = F.relu(output_bn)
		#output = torch.clamp(output_bn, min=0, max=T)

		# extract the weight and bias from the surrogate conv layer
		convTranspose2d_weight = self.convTranspose2d.weight#.detach().to(self.device)
		convTranspose2d_bias = self.convTranspose2d.bias#.detach().to(self.device)
		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights

		ratio = torch.div(bnGamma, torch.sqrt(bnVar))

		weightNorm = torch.mul(convTranspose2d_weight.permute(0,2,3,1), ratio).permute(0,3,1,2)
		biasNorm = torch.mul(convTranspose2d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.convTranspose2dIF(input_feature_st, output,\
														weightNorm, self.device, biasNorm,\
														self.stride, self.padding, self.output_padding,self.pooling)

		return output_features_st, output_features_sc

class ConvBN2d(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0, bias=True, weight_init=2.0, pooling=1):

		super(ConvBN2d, self).__init__()
		self.conv2dIF = Conv2dIF.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
		# self.bn2d = torch.nn.BatchNorm2d(Cout)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling

		nn.init.normal_(self.bn2d.weight, 0, weight_init)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate conv2d layer
		output = F.relu(F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling))

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)
		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))

		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2)
		biasNorm = torch.mul(conv2d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output,\
														weightNorm, self.device, biasNorm,\
														self.stride, self.padding, self.pooling)

		return output_features_st, output_features_sc

class LinearBN1d(nn.Module):

	def __init__(self, D_in, D_out, device=torch.device(SETTINGS.training.device), bias=True):
		super(LinearBN1d, self).__init__()
		self.linearif = LinearIF.apply
		self.device = device
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		# self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
		self.bn1d = torch.nn.BatchNorm1d(D_out,eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
		# nn.init.normal_(self.bn1d.weight, 0, 2.0)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		# print(input_features_sc.shape)
		output_bn = self.bn1d(self.linear(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight#.detach().to(self.device)
		linearif_bias = self.linear.bias#.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
		biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, weightNorm,  \
												self.device, biasNorm)

		return output_st, output_sc

####################################################
# class OperationFCBN1d(nn.Module):
#
# 	def __init__(self, D_in, D_out, device=torch.device(SETTINGS.training.device), bias=True,momentum =0.5,eps=1e-4): # momentum =0.5
# 		super(OperationFCBN1d, self).__init__()
# 		self.linearif =LinearIF.apply #OperationFCIF.apply
# 		self.device = device
# 		#self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
# 		self.linear = OperationFC(D_in, D_out, bias=bias)
# 		# self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-6, momentum=0.9)
# 		self.bn1d = torch.nn.BatchNorm1d(D_out,affine=True)#,momentum=momentum,eps=eps
# 		# nn.init.normal_(self.bn1d.weight, 0, 2)#2.0
# 		# nn.init.normal_(self.bn1d.bias, -0.1, 0.1)
# 		# nn.init.normal_(self.bn1d.weight, 0.5, 0.5)#2.0
#
# 	def forward(self, input_feature_st, input_features_sc):
# 		# weight update based on the surrogate linear layer
# 		output = F.relu(self.bn1d(self.linear(input_features_sc)))
# 		# extract the weight and bias from the surrogate linear layer
#
# 		# re-parameterization by integrating the beta and gamma factors
# 		# into the 'Linear' layer weights
# 		ratio = torch.div(self.bn1d.weight, torch.sqrt(self.bn1d.running_var))#+self.bn1d.eps
# 		weightNorm = torch.mul( self.linear.weights.permute(1, 0), ratio).permute(1, 0)
# 		biasNorm = torch.mul(self.linear.bias-self.bn1d.running_mean, ratio) + self.bn1d.bias
#
# 		# propagate the input spike train through the linearIF layer to get actual output
# 		# spike train
# 		output_st, output_sc = self.linearif(input_feature_st, output, weightNorm, self.device, biasNorm)
#
# 		return output_st, output_sc
#
# class OperationConvBN2d(nn.Module):
# 	"""
# 	W
# 	"""
# 	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0, bias=True, weight_init=2.0, pooling=1,momentum=0.9,eps=1e-5): #momentum =0.9
#
# 		super(OperationConvBN2d, self).__init__()
# 		self.conv2dIF = Conv2dIF.apply#OperationConv2dIF.apply
# 		self.conv2d = OperationConv2d(Cin, Cout,kernel_size, stride,padding, bias)#torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
# 		#self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
# 		# self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
# 		self.bn2d = torch.nn.BatchNorm2d(Cout,affine=True)#,momentum=momentum,eps=eps
# 		self.device = device
# 		self.stride = stride
# 		self.padding = padding
# 		self.pooling = pooling
# 		# nn.init.normal_(self.bn2d.weight, 0.5, 0.5)
# 		# nn.init.normal_(self.bn2d.weight, 0, 2)
# 		# nn.init.normal_(self.bn2d.bias, -0.5, 0.35)
#
# 	def forward(self, input_feature_st, input_features_sc):
#
# 		# weight update based on the surrogate conv2d layer
# 		output = F.relu(F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling))
#
# 		# extract the weight and bias from the surrogate conv layer
#
# 		# re-parameterization by integrating the beta and gamma factors
# 		# into the 'Conv' layer weights
# 		# ratio = torch.div(self.bn2d.weight, torch.sqrt(self.bn2d.running_var))
# 		# weightNorm = torch.mul(self.conv2d.weights.permute(1,2,3,0), ratio).permute(3,0,1,2)
# 		# biasNorm = torch.mul(self.conv2d.bias-self.bn2d.running_mean, ratio) + self.bn2d.bias
#
# 		ratio = torch.div(self.bn2d.weight,torch.sqrt(self.bn2d.running_var))#+self.bn2d.eps
# 		weightNorm = torch.mul(self.conv2d.weights.permute(1,2,3,0), ratio).permute(3,0,1,2)
# 		biasNorm = torch.mul(self.conv2d.bias-self.bn2d.running_mean, ratio) + self.bn2d.bias
# 		# propagate the input spike train through the IF layer to get actual output
# 		# spike train
# 		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, weightNorm, self.device,
# 															biasNorm, self.stride, self.padding, self.pooling)
#
# 		return output_features_st, output_features_sc

###############################################


##################################################











class sDropout(nn.Module):
	def __init__(self, layerType, pDrop):
		super(sDropout, self).__init__()

		self.pKeep = 1 - pDrop
		self.type = layerType # 1: Linear 2: Conv

	def forward(self, x_st, x_sc):
		if self.training:
			T = x_st.shape[1]
			mask = torch.bernoulli(x_sc.data.new(x_sc.data.size()).fill_(self.pKeep))/self.pKeep
			x_sc_out = x_sc * mask
			x_st_out = torch.zeros_like(x_st)
			
			for t in range(T):
				# Linear Layer
				if self.type == 1:
					x_st_out[:,t,:] = x_st[:,t,:] * mask
				# Conv1D Layer
				elif self.type == 2:
					x_st_out[:,t,:,:] = x_st[:,t,:,:] * mask
				# Conv2D Layer					
				elif self.type == 3:
					x_st_out[:,t,:,:,:] = x_st[:,t,:,:,:] * mask
		else:					
			x_sc_out = x_sc
			x_st_out = x_st
			
		return x_st_out, x_sc_out
		
class Linear(nn.Module):

	def __init__(self, D_in, D_out, net_params, device=torch.device('cpu'), bias=True):
		super(Linear, self).__init__()

		self.net_params = net_params
		self.linearif = LinearIF.apply
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.device = device

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_round = torch.floor(self.linear(input_features_sc))
		output = torch.clamp(output_round, min=0, max=T)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight.detach().to(self.device)
		linearif_bias = self.linear.bias.detach().to(self.device)

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, linearif_weight, self.net_params, \
												self.device, linearif_bias)

		return output_st, output_sc

class LinearDropout1d(nn.Module):

	def __init__(self, D_in, D_out, device=torch.device('cuda:0'), bias=True,drop=0.5):
		super(LinearDropout1d, self).__init__()
		self.linearif = LinearIF.apply
		self.device = device
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.dropout = nn.Dropout(drop)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_bn = self.dropout(self.linear(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight#.detach().to(self.device)
		linearif_bias = self.linear.bias#.detach().to(self.device)

		# bnGamma = self.bn1d.weight
		# bnBeta = self.bn1d.bias
		# bnMean = self.bn1d.running_mean
		# bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		# ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		# weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
		# biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, linearif_weight,  \
												self.device, linearif_bias)

		return output_st, output_sc


class ConvBN1d(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, kernel_size, device=torch.device('cpu'), stride=1, \
					padding=0, bias=True, weight_init=2.0):

		super(ConvBN1d, self).__init__()
		self.conv1dIF = Conv1dIF.apply
		self.conv1d = torch.nn.Conv1d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		self.bn1d = torch.nn.BatchNorm1d(Cout, eps=1e-4, momentum=0.9)
		self.device = device
		self.stride = stride
		self.padding = padding
		nn.init.normal_(self.bn1d.weight, 0, weight_init)

	def forward(self, input_feature_st, input_features_sc):
		T = input_feature_st.shape[1]
		
		# weight update based on the surrogate conv2d layer
		output_bn = self.bn1d(self.conv1d(input_features_sc))
		output = F.relu(output_bn)
		#output = torch.clamp(output_bn, min=0, max=T)

		# extract the weight and bias from the surrogate conv layer
		conv1d_weight = self.conv1d.weight.detach().to(self.device)
		conv1d_bias = self.conv1d.bias.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv1d_weight.permute(1,2,0), ratio).permute(2,0,1) 
		biasNorm = torch.mul(conv1d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv1dIF(input_feature_st, output,\
														weightNorm, self.device,\
														biasNorm, self.stride, self.padding)

		return output_features_st, output_features_sc



#class Conv2d(nn.Module):
#
#	def __init__(self, Cin, Cout, kernel_size, net_params, layerIdx, device=torch.device('cpu'), stride=1, padding=0, bias=True):
#		super(Conv2d, self).__init__()
#
#		self.net_params = net_params
#		self.conv2dIF = Conv2dIF.apply
#		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
#		self.device = device
#		self.stride = stride
#		self.padding = padding
#		self.layerIdx = layerIdx
#
#	def forward(self, input_feature_spikes, input_features):
#		# extract the weight and bias from the surrogate conv2d layer
#		conv2dif_weight = self.conv2d.weight.detach().cpu()
#		self.conv2d.bias.data.zero_()
#		conv2dif_bias = self.conv2d.bias.detach().cpu()
#
#		# propagate the input spike train through the linearIF layer to get actual output
#		# spike train
#		output_features_spikes, input_spike_count = self.conv2dIF(self.layerIdx, input_feature_spikes, input_features,\
#														conv2dif_weight, self.net_params, self.device,\
#														conv2dif_bias, self.stride, self.padding)
#
#		# weight update based on the surrogate conv2d layer
#		output_feature = self.conv2d(input_spike_count)
#
#		return output_features_spikes, output_feature
#
#class AvgPool2d(nn.Module):
#	"""
#		Wrapper for average pooling layer and spiking average pooling layer (IF neurons)
#	"""
#	def __init__(self, kernel_size, net_params, layerIdx, stride=None, padding=0, device=torch.device('cpu')):
#		super(AvgPool2d, self).__init__()
#
#		self.net_params = net_params
#		self.avg_pool2d = AvgPool2d_IF.apply
#		self.kernel_size = kernel_size
#		self.stride = stride
#		self.padding = padding
#		self.device = device
#		self.layerIdx = layerIdx
#		self.floor = floor.apply
#
#	def forward(self, input_feature_spikes, input_features):
#
#		# propagate the input spike train through the AvgPool2d_IF layer to get actual output
#		# spike train
#		output_features_spikes, input_spike_count = self.avg_pool2d(self.layerIdx, input_feature_spikes, input_features,\
#														self.net_params, self.kernel_size, self.stride,\
#														self.padding, self.device)
#
#		output_feature = self.floor(F.avg_pool2d(input_spike_count, self.kernel_size, self.stride, self.padding))
#
#		return output_features_spikes, output_feature
#
#
#
#class MaxPool2d(nn.Module):
#	"""
#		Wrapper for average pooling layer and spiking average pooling layer (IF neurons)
#	"""
#	def __init__(self, kernel_size, net_params, layerIdx, stride=None, padding=0, device=torch.device('cpu')):
#		super(AvgPool2d, self).__init__()
#
#		self.net_params = net_params
#		self.max_pool2d = MaxPool2d_IF.apply
#		self.kernel_size = kernel_size
#		self.stride = stride
#		self.padding = padding
#		self.device = device
#		self.layerIdx = layerIdx
#		self.floor = floor.apply
#
#	def forward(self, input_feature_spikes, input_features):
#
#		# propagate the input spike train through the AvgPool2d_IF layer to get actual output
#		# spike train
#		output_features_spikes, input_spike_count = self.max_pool2d(self.layerIdx, input_feature_spikes, input_features,\
#														self.net_params, self.kernel_size, self.stride,\
#														self.padding, self.device)
#
#		output_feature = self.floor(F.max_pool2d(input_spike_count, self.kernel_size, self.stride, self.padding))
#
#		return output_features_spikes, output_feature

################################################################################################

class actF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, pot_aggregate,spike_mask):
		"""
		args:

		"""

		spike=pot_aggregate.ge(1.0).float()
		spike *= (1 - spike_mask)
		ctx.save_for_backward(spike)
		# print(feature_out.requires_grad)

		return spike

	@staticmethod
	def backward(ctx, grad_feature_out):
		spike, = ctx.saved_tensors
		grad_feature_in = grad_feature_out.clone()

		return grad_feature_in * (-spike), None





actF = actF.apply
class ConvBN2d_(nn.Module):
	"""
	W
	"""

	def __init__(self, Cin, Cout, kernel_size, device=torch.device('cuda:2'), stride=1, padding=0, bias=True,
				 weight_init=2.0, pooling=1):
		super(ConvBN2d_, self).__init__()
		self.conv2dIF = Conv2dIF.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		# self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
		self.bn2d = torch.nn.BatchNorm2d(Cout)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling
		nn.init.normal_(self.bn2d.weight, 0, weight_init)

	def forward(self, input_feature_st, input_features_sc):


		# weight update based on the surrogate conv2d layer
		output_bn = F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling)
		ann_output = F.relu(output_bn)
		# output = torch.clamp(output_bn, min=0, max=T)

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1, 2, 3, 0), ratio).permute(3, 0, 1, 2)
		biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		# output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, \
		# 													   weightNorm, self.device, biasNorm, \
		# 													   self.stride, self.padding, self.pooling)
		N, T, in_channels, iH, iW = input_feature_st.shape
		out_channels, in_channels, kH, kW = weightNorm.shape
		pot_aggregate = torch.zeros_like(output_bn) # init the membrane potential with the bias
		_, _, outH, outW = pot_aggregate.shape
		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=self.device)
		spike_mask = torch.zeros_like(pot_aggregate, device=self.device).float()
		spike_count_out=torch.zeros_like(spike_out[:,0,:,:,:])

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += F.max_pool2d(F.conv2d(input_feature_st[:, t, :, :, :], weightNorm, biasNorm, self.stride, self.padding), self.pooling)
			bool_spike = actF(pot_aggregate,spike_mask)

			spike_count_out +=bool_spike
			spike_out[:, t, :, :, :] = bool_spike
			pot_aggregate -= bool_spike

			spike_mask += bool_spike
			spike_mask[spike_mask > 0] = 1

		return spike_out, spike_count_out, ann_output




class LinearBN1d_(nn.Module):

	def __init__(self, D_in, D_out, device=torch.device('cuda:2'), bias=True):
		super(LinearBN1d_, self).__init__()
		self.linearif = LinearIF.apply
		self.device = device
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		# self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
		self.bn1d = torch.nn.BatchNorm1d(D_out)
		nn.init.normal_(self.bn1d.weight, 0, 2.0)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_bn = self.bn1d(self.linear(input_features_sc))
		ann_output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight#.detach().to(self.device)
		linearif_bias = self.linear.bias#.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
		biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		# output_st, output_sc = self.linearif(input_feature_st, output, weightNorm,  \
		# 										self.device, biasNorm)
		N, T, _ = input_feature_st.shape
		pot_in = input_feature_st.matmul(weightNorm.t())
		spike_out = torch.zeros_like(pot_in, device=self.device)
		pot_aggregate = biasNorm.repeat(N, 1)  # init the membrane potential with the bias
		spike_mask = torch.zeros_like(pot_aggregate, device=self.device).float()
		spike_count_out=torch.zeros_like(spike_out[:,0,:])

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in[:, t, :].squeeze()
			bool_spike = actF(pot_aggregate,spike_mask)
			spike_count_out +=bool_spike
			spike_out[:, t, :] = bool_spike
			pot_aggregate -= bool_spike

			spike_mask += bool_spike
			spike_mask[spike_mask > 0] = 1

		return spike_out, spike_count_out, ann_output
