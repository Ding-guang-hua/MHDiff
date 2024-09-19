import torch
# 导入 PyTorch 库，PyTorch 是一个用于机器学习和深度学习的开源框架

from torch import nn
# 从 PyTorch 库中导入 neural network 模块（nn），用于构建神经网络

import torch.nn.functional as F
# 导入 PyTorch 中的函数库（functional），通常用于定义网络的操作

from Params import args
# 从 Params 模块中导入参数 args，用于设置和配置模型的参数

import numpy as np
# 导入 NumPy 库并将其重命名为 np，用于处理多维数组和矩阵运算
import random
# 导入 Python 的 random 库，用于生成随机数

import math
# 导入 Python 的 math 库，包含数学函数

from Utils.Utils import *
# 从 Utils 文件夹中的 Utils 模块导入所有内容，这里应包含一些自定义的工具函数

init = nn.init.xavier_uniform_
# xavier_uniform_ 是一种 Xavier 初始化器的变种，用于将权重初始化为服从均匀分布的随机值
# 会修改传入的权重张量（tensor）以进行初始化，不返回新的张量，而是直接对传入的参数进行修改

uniformInit = nn.init.uniform
# 对权重张量进行初始化，使其从一个均匀分布中抽取初始值

class Model(nn.Module):
	# 定义了一个名为 Model 的类，继承自 nn.Module，用于构建一个神经网络模型
	def __init__(self, image_embedding, text_embedding, image_id, text_id, audio_embedding=None, audio_id=None):
		# 定义了 Model 类的初始化函数，接受图像嵌入、文本嵌入和音频嵌入作为参数，音频嵌入参数为可选参数
		super(Model, self).__init__()
		# 调用父类 nn.Module 的初始化方法

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		# 初始化用户嵌入矩阵 uEmbeds，参数为用户数和潜在特征维度
		# torch.empty(args.user, args.latdim) 创建了一个指定大小（args.user 行 args.latdim 列）的空张量
		# nn.Parameter() 会将初始化后的张量转化为可训练的参数，使得这个张量可以被模型训练过程中更新权重参数。
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		# 初始化物品嵌入矩阵 iEmbeds，参数为物品数和潜在特征维度
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		# 创建一系列的 GCNLayer 层，用于图卷积网络的构建

		self.edgeDropper = SpAdjDropEdge(args.keepRate)
		# 初始化边缘丢弃器，参数为保留概率

		if args.trans == 1: #default=0，0: R*R, 1: Linear, 2: allrecipes   latdim=64,
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.image_id_trans = nn.Linear(args.image_id_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
			self.text_id_trans = nn.Linear(args.text_id_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.image_id_trans = nn.Parameter(init(torch.empty(size=(args.image_id_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
			self.text_id_trans = nn.Parameter(init(torch.empty(size=(args.text_id_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.image_id_trans = nn.Parameter(init(torch.empty(size=(args.image_id_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
			self.text_id_trans = nn.Linear(args.text_id_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
				self.audio_id_trans = nn.Linear(args.audio_id_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))
				self.audio_id_trans = nn.Parameter(init(torch.empty(size=(args.audio_id_dim, args.latdim))))
		# 根据参数 args.trans 的值，初始化图像和文本的转换层，根据不同的情况选择不同的初始化方式
		# 根据是否有音频嵌入，初始化相应的音频转换层，根据 args.trans 的不同值选择不同的初始化方法


		self.image_embedding = image_embedding
		self.image_id = image_id
		# 将输入的图像嵌入赋值给类中的图像嵌入属性
		self.text_embedding = text_embedding
		self.text_id = text_id
		# 将输入的文本嵌入赋值给类中的文本嵌入属性
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
			self.audio_id = audio_id
		else:
			self.audio_embedding = None
			self.audio_id = None
		# 根据是否存在音频嵌入，初始化类中的音频嵌入属性

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		# 根据是否存在音频嵌入，初始化模态权重，用于组合不同媒体的信息
		self.softmax = nn.Softmax(dim=0)
		# 定义在维度 0 上进行 softmax 操作的 Softmax 层，通常是 Batch 维度

		self.dropout = nn.Dropout(p=0.1)
		# 定义概率为 0.1 的 Dropout 层，用于防止过拟合，  按照 10% 的概率随机丢弃每个输入元素，即置零

		self.leakyrelu = nn.LeakyReLU(0.2)
	# 	定义斜率为 0.2 的 LeakyReLU 激活函数，在 LeakyReLU 函数中当输入为负时的斜率为 0.2，使得负数部分的信号不会被完全抑制
				
	def getItemEmbeds(self):
		return self.iEmbeds
	# 定义函数 getItemEmbeds ，函数返回存储物品嵌入参数的 iEmbeds 属性，允许模型获取物品的嵌入表示
	
	def getUserEmbeds(self):
		return self.uEmbeds
	# 定义函数 getUserEmbeds ，函数返回存储用户嵌入参数的 uEmbeds 属性，允许模型获取用户的嵌入表示

	def getAllFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
		else:
			image_feats = self.image_trans(self.image_embedding)
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		else:
			text_feats = self.text_trans(self.text_embedding)
		if self.audio_embedding != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		weight = self.softmax(self.modal_weight)
		if self.audio_embedding != None:
			allFeats = weight[0] * image_feats + weight[1] * text_feats + weight[2] * audio_feats
		else:
			allFeats = weight[0] * image_feats + weight[1] * text_feats
		return allFeats

	def getAllID(self):
		if args.trans == 0 or args.trans == 2:
			imageID = self.leakyrelu(torch.mm(self.image_id, self.image_id_trans))
		else:
			imageID = self.image_id_trans(self.image_id)
		if args.trans == 0:
			textID = self.leakyrelu(torch.mm(self.text_id, self.text_id_trans))
		else:
			textID = self.text_id_trans(self.text_id)
		if self.audio_id != None:
			if args.trans == 0:
				audioID = self.leakyrelu(torch.mm(self.audio_id, self.audio_id_trans))
			else:
				audioID = self.audio_id_trans(self.audio_id)
		weight = self.softmax(self.modal_weight)
		if self.audio_id != None:
			allID = weight[0] * imageID + weight[1] * textID + weight[2] * audioID
		else:
			allID = weight[0] * imageID + weight[1] * textID
		return allID

	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	# 函数根据条件判断选择不同的处理方式：
	# 若转换类型为0或2，则通过矩阵乘法和 LeakyReLU 激活函数计算输入图像和图像转换矩阵之间的特征，返回处理后的图像特征。
	# 否则，直接对输入的图像进行转换操作并返回转换后的特征。
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)
	# 函数根据条件判断选择不同的处理方式：
	# 若转换类型为0，则通过矩阵乘法和 LeakyReLU 激活函数计算输入文本和文本转换矩阵之间的特征，返回处理后的文本特征。
	# 否则，直接对输入的文本进行转换操作并返回转换后的特征。

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats
	# 函数根据条件判断选择不同的处理方式：
	# 如果没有音频嵌入，则返回空值。
	# 如果存在音频嵌入：
	# 若转换类型为0，则通过矩阵乘法和 LeakyReLU 激活函数计算输入音频和音频转换矩阵之间的特征，并返回处理后的音频特征。
	# 否则，直接对输入的音频进行转换操作并返回转换后的特征。

# 以上这些函数允许模型存取各媒体类型（图像、文本、音频）的特征表示，根据指定的转换类型执行相应的处理操作以获取最终特征表示


	def forward_Main(self, adj, id_adj, feature_adj):
		# 定义了一个名为 forward_MM 的函数，这个函数接受四个参数：adj、image_adj、text_adj 和可选的 audio_adj
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			imageID = self.leakyrelu(torch.mm(self.image_id, self.image_id_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			textID = self.leakyrelu(torch.mm(self.text_id, self.text_id_trans))
		# 判断 args.trans 的值是否为 0，如果是，对图像和文本嵌入进行矩阵乘法运算，
		# 然后应用 LeakyReLU 激活函数得到 image_feats 和 text_feats
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			imageID = self.image_id_trans(self.image_id)
			text_feats = self.text_trans(self.text_embedding)
			textID = self.text_id_trans(self.text_id)
		# 如果 args.trans 的值为 1，直接使用转换函数对图像和文本嵌入进行处理，得到 image_feats 和 text_feats
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			imageID = self.leakyrelu(torch.mm(self.image_id, self.image_id_trans))
			text_feats = self.text_trans(self.text_embedding)
			textID = self.text_id_trans(self.text_id)
		# 在其他情况下，对图像使用矩阵乘法和 LeakyReLU 激活函数，对文本使用转换函数，得到 image_feats 和 text_feats
		if self.audio_embedding != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
				audioID = self.leakyrelu(torch.mm(self.audio_id, self.audio_id_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
				audioID = self.audio_id_trans(self.audio_id)
		# 如果存在音频邻接矩阵 (audio_adj)，则检查 args.trans 的取值。如果为 0，
		# 则对音频嵌入进行矩阵乘法和 LeakyReLU 激活函数计算得到 audio_feats；否则直接使用转换函数处理音频嵌入

		weight = self.softmax(self.modal_weight)
		# 计算模态权重，通过 softmax 函数对模态权重进行归一化。
		if self.audio_embedding != None:
			allFeats = weight[0] * image_feats + weight[1] * text_feats + weight[2] * audio_feats
			allID = weight[0] * imageID + weight[1] * textID + weight[2] * audioID
		else:
			allFeats = weight[0] * image_feats + weight[1] * text_feats
			allID = weight[0] * imageID + weight[1] * textID



		embedsIdAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsIdAdj = torch.spmm(id_adj, embedsIdAdj)
		# 将用户嵌入和物品嵌入连接起来，并对图像邻接矩阵进行稀疏矩阵乘法操作，结果存储在 embedsImageAdj 中

		embedsId = torch.concat([self.uEmbeds, F.normalize(allID)])
		embedsId = torch.spmm(adj, embedsId)
		# 将用户嵌入和经过归一化处理后的图像特征连接起来，然后对整体进行稀疏矩阵乘法操作，将结果存储在 embedsImage 中

		embedsId_ = torch.concat([embedsId[:args.user], self.iEmbeds])
		embedsId_ = torch.spmm(adj, embedsId_)
		embedsId += embedsId_
		# 将之前得到的 embedsImage 切片（取前 args.user 个元素），
		# 然后与物品嵌入连接，对其进行稀疏矩阵乘法操作，并将结果加到原始的 embedsImage 中

		embedsFeatAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsFeatAdj = torch.spmm(feature_adj, embedsFeatAdj)
		# 类似于处理图像数据，将用户嵌入和物品嵌入连接起来，并对文本邻接矩阵进行稀疏矩阵乘法操作，
		# 结果存储在 embedsTextAdj 中

		embedsFeat = torch.concat([self.uEmbeds, F.normalize(allFeats)])
		embedsFeat = torch.spmm(adj, embedsFeat)
		# 将用户嵌入和经过归一化处理后的文本特征连接起来，然后对整体进行稀疏矩阵乘法操作，将结果存储在 embedsText 中
		# 这部分代码处理了文本数据的特征

		embedsFeat_ = torch.concat([embedsFeat[:args.user], self.iEmbeds])
		embedsFeat_ = torch.spmm(adj, embedsFeat_)
		embedsFeat += embedsFeat_
		# 类似于处理图像数据，将之前得到的 embedsText 切片（取前 args.user 个元素），
		# 然后与物品嵌入连接，对其进行稀疏矩阵乘法操作，并将结果加到原始的 embedsText 中

		embedsId += args.ris_adj_lambda * embedsIdAdj
		# 对图像嵌入特征进行加权处理，加权值为 args.ris_adj_lambda，
		# 并将图像特征邻接矩阵的乘积结果 embedsImageAdj 加到原始图像嵌入特征 embedsImage 上
		embedsFeat += args.ris_adj_lambda * embedsFeatAdj
		# 对文本嵌入特征进行加权处理，加权值为 args.ris_adj_lambda，
		# 并将文本特征邻接矩阵的乘积结果 embedsTextAdj 加到原始文本嵌入特征 embedsText 上
		embedsModal = 0.5 * embedsId+ 0.5 * embedsFeat

		embeds = embedsModal
		# 将模态融合后的整体嵌入特征赋值给 embeds
		embedsLst = [embeds]
		# 循环遍历每个 GCN 层
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			# 对当前嵌入特征和上一层嵌入特征进行 GCN 操作，得到新的嵌入特征
			embedsLst.append(embeds)
		# 将新的嵌入特征添加到嵌入特征列表中
		embeds = sum(embedsLst)
		# 将所有 GCN 层的嵌入特征进行累加求和得到最终的嵌入特征 embeds

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)
		# 将最终嵌入特征 embeds 与经过归一化处理的整体嵌入特征 embedsModal 加权相加

		return embeds[:args.user], embeds[args.user:]


	#主视图（多模态图聚合）
	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
	# 定义了一个名为 forward_MM 的函数，这个函数接受四个参数：adj、image_adj、text_adj 和可选的 audio_adj
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		# 判断 args.trans 的值是否为 0，如果是，对图像和文本嵌入进行矩阵乘法运算，
		# 然后应用 LeakyReLU 激活函数得到 image_feats 和 text_feats
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		# 如果 args.trans 的值为 1，直接使用转换函数对图像和文本嵌入进行处理，得到 image_feats 和 text_feats
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)
		# 在其他情况下，对图像使用矩阵乘法和 LeakyReLU 激活函数，对文本使用转换函数，得到 image_feats 和 text_feats

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		# 如果存在音频邻接矩阵 (audio_adj)，则检查 args.trans 的取值。如果为 0，
	    # 则对音频嵌入进行矩阵乘法和 LeakyReLU 激活函数计算得到 audio_feats；否则直接使用转换函数处理音频嵌入

		weight = self.softmax(self.modal_weight)
	    # 计算模态权重，通过 softmax 函数对模态权重进行归一化。

		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)
	    # 将用户嵌入和物品嵌入连接起来，并对图像邻接矩阵进行稀疏矩阵乘法操作，结果存储在 embedsImageAdj 中

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)
	    # 将用户嵌入和经过归一化处理后的图像特征连接起来，然后对整体进行稀疏矩阵乘法操作，将结果存储在 embedsImage 中

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_
	    # 将之前得到的 embedsImage 切片（取前 args.user 个元素），
	    # 然后与物品嵌入连接，对其进行稀疏矩阵乘法操作，并将结果加到原始的 embedsImage 中
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)
	    # 类似于处理图像数据，将用户嵌入和物品嵌入连接起来，并对文本邻接矩阵进行稀疏矩阵乘法操作，
	    # 结果存储在 embedsTextAdj 中

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)
		# 将用户嵌入和经过归一化处理后的文本特征连接起来，然后对整体进行稀疏矩阵乘法操作，将结果存储在 embedsText 中
	# 这部分代码处理了文本数据的特征

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_
		# 类似于处理图像数据，将之前得到的 embedsText 切片（取前 args.user 个元素），
	    # 然后与物品嵌入连接，对其进行稀疏矩阵乘法操作，并将结果加到原始的 embedsText 中

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)
			# 如果存在音频邻接矩阵，将用户嵌入和物品嵌入连接起来，并对音频邻接矩阵进行稀疏矩阵乘法操作，
			# 结果存储在 embedsAudioAdj 中

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)
			# 将用户嵌入和经过归一化处理后的音频特征连接起来，然后对整体进行稀疏矩阵乘法操作，
			# 将结果存储在 embedsAudio 中
		# 这部分代码处理了音频数据的特征

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_
		# 类似于处理图像和文本数据，针对音频数据，将之前得到的 embedsAudio 切片（取前 args.user 个元素），
	    # 然后与物品嵌入连接，对其进行稀疏矩阵乘法操作，并将结果加到原始的 embedsAudio 中

		embedsImage += args.ris_adj_lambda * embedsImageAdj
		# 对图像嵌入特征进行加权处理，加权值为 args.ris_adj_lambda，
	    # 并将图像特征邻接矩阵的乘积结果 embedsImageAdj 加到原始图像嵌入特征 embedsImage 上
		embedsText += args.ris_adj_lambda * embedsTextAdj
		# 对文本嵌入特征进行加权处理，加权值为 args.ris_adj_lambda，
	    # 并将文本特征邻接矩阵的乘积结果 embedsTextAdj 加到原始文本嵌入特征 embedsText 上
		if audio_adj != None:
			# 检查是否存在音频邻接矩阵
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
	    # 若存在音频邻接矩阵，则对音频嵌入特征进行加权处理，加权值为 args.ris_adj_lambda，
	    # 并将音频特征邻接矩阵的乘积结果 embedsAudioAdj 加到原始音频嵌入特征 embedsAudio 上

		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		# 如果不存在音频邻接矩阵，则计算模态融合，采用图像和文本的权重相加作为整体嵌入特征 embedsModal
		# 使用图像和文本的权重相加来计算整体嵌入特征 embedsModal
		else:
			# 如果存在音频邻接矩阵，则将音频嵌入特征也考虑在内进行模态融合
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio
		# 加权计算整体的嵌入特征 embedsModal，包含图像、文本和音频



		embeds = embedsModal
		# 将模态融合后的整体嵌入特征赋值给 embeds
		embedsLst = [embeds]
		# 循环遍历每个 GCN 层
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			# 对当前嵌入特征和上一层嵌入特征进行 GCN 操作，得到新的嵌入特征
			embedsLst.append(embeds)
		    # 将新的嵌入特征添加到嵌入特征列表中
		embeds = sum(embedsLst)
	    # 将所有 GCN 层的嵌入特征进行累加求和得到最终的嵌入特征 embeds

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)
	    # 将最终嵌入特征 embeds 与经过归一化处理的整体嵌入特征 embedsModal 加权相加

		return embeds[:args.user], embeds[args.user:]
	# 返回给用户和物品的分别的嵌入特征

	def forward_CL(self, adj, id_adj, feature_adj, all_feats, all_id):

		embedsFeature = torch.concat([self.uEmbeds, F.normalize(all_feats)])
		# 将用户嵌入（self.uEmbeds）与归一化的图像特征（image_feats）连接起来，以生成图像的嵌入表示。E
		embedsFeature = torch.spmm(feature_adj, embedsFeature)
		embedsId = torch.concat([self.uEmbeds, F.normalize(all_id)])
		# 将用户嵌入（self.uEmbeds）与归一化的图像特征（image_feats）连接起来，以生成图像的嵌入表示。E
		embedsId = torch.spmm(feature_adj, embedsId)

		# embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		# # 将用户嵌入（self.uEmbeds）与归一化的文本特征（text_feats）连接起来，以生成文本的嵌入表示。
		# embedsText = torch.spmm(text_adj, embedsText)
		# # 使用稀疏矩阵-矩阵乘法（Sparse Matrix Multiplication）操作，
		# # 将文本邻接矩阵（text_adj）与文本嵌入表示（embedsText）相乘，得到更新后的文本嵌入表示。
		# # 在这段代码中，除了处理音频特征外，它还将图像和文本特征与用户嵌入连接起来，并通过稀疏矩阵的乘法操作更新图像和文本的嵌入表示。
		# # 这些操作通常用于将不同类型的特征结合起来以进行后续的模型训练或预测。
		#
		# if audio_adj != None:
		# 	# 如果存在音频邻接矩阵（audio_adj）：
		# 	embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
		# 	# 将用户嵌入（self.uEmbeds）与归一化的音频特征（audio_feats）连接起来，生成音频的嵌入表示。
		# 	embedsAudio = torch.spmm(audio_adj, embedsAudio)
		# # 	使用稀疏矩阵-矩阵乘法（Sparse Matrix Multiplication）操作，
		# # 	将音频邻接矩阵（audio_adj）与音频嵌入表示（embedsAudio）相乘，得到更新后的音频嵌入表示。

		embeds1 = embedsId
		# 把图像嵌入（embedsImage）赋值给embeds1。
		embedsLst1 = [embeds1]
		# 创建一个列表embedsLst1，并将embeds1添加到该列表中。
		for gcn in self.gcnLayers:
			# 对于self.gcnLayers中的每个GCN层：
			embeds1 = gcn(adj, embedsLst1[-1])#Zt
			# 使用当前GCN层（gcn）对邻接矩阵（adj）和前一层的输出（embedsLst1中的最后一个元素）进行操作，得到更新后的embeds1。
			embedsLst1.append(embeds1)
		# 	将更新后的embeds1添加到embedsLst1列表中。
		embeds1 = sum(embedsLst1)
		# 对embedsLst1列表中的所有元素求和，得到最终的embeds1表示。

		embeds2 = embedsFeature
		# 把文本嵌入（embedsText）赋值给embeds2。
		embedsLst2 = [embeds2]
		# 创建一个列表embedsLst2，并将embeds2添加到该列表中。
		for gcn in self.gcnLayers:
			# 对于self.gcnLayers中的每个GCN层：
			embeds2 = gcn(adj, embedsLst2[-1])
			# 使用当前GCN层（gcn）对邻接矩阵（adj）和前一层的输出（embedsLst2中的最后一个元素）进行操作，得到更新后的embeds2。
			embedsLst2.append(embeds2)
		# 	将更新后的embeds2添加到embedsLst2列表中。
		embeds2 = sum(embedsLst2)
		# 对embedsLst2列表中的所有元素求和，得到最终的embeds2表示。
		# 这段代码主要在处理嵌入表示的计算，包括将不同类型的嵌入信息与邻接矩阵结合，
		# 通过GCN（图卷积网络）层来更新和整合不同类型的嵌入表示，最终得到综合的图像和文本的嵌入表示。
		# 这些表示通常用于后续的图神经网络任务。

		return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		# 返回图像和文本嵌入表示中针对用户（args.user）进行分割后的结果。

	# 这个代码片段是一个模型的前向传播函数中的一部分，
	# 接收多个输入邻接矩阵（adj、image_adj、text_adj、audio_adj）并基于条件（args.trans）进行操作。
	# 以下是对每行代码的中文详细分析：
	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			# 如果参数 args.trans 的值为0：
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			# 计算图像特征：将图像嵌入（self.image_embedding）与图像变换矩阵（self.image_trans）相乘，
			# 然后通过 LeakyReLU 激活函数处理得到图像特征。
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		# 	计算文本特征：将文本嵌入（self.text_embedding）与文本变换矩阵（self.text_trans）相乘，
		# 	然后通过 LeakyReLU 激活函数处理得到文本特征。
		elif args.trans == 1:
			# 否则，如果参数 args.trans 的值为1：
			image_feats = self.image_trans(self.image_embedding)
			# 对图像嵌入（self.image_embedding）进行图像转换函数（self.image_trans）操作得到图像特征。
			text_feats = self.text_trans(self.text_embedding)
		# 	对文本嵌入（self.text_embedding）进行文本转换函数（self.text_trans）操作得到文本特征。
		else:
			# 否则：
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			# 计算图像特征：将图像嵌入（self.image_embedding）与图像变换矩阵（self.image_trans）相乘，
			# 然后通过 LeakyReLU 激活函数处理得到图像特征。
			text_feats = self.text_trans(self.text_embedding)
		# 	对文本嵌入（self.text_embedding）进行文本转换函数（self.text_trans）操作得到文本特征。

		# 这段代码继续模型的前向传播，处理音频特征（如果有），并针对图像和文本特征进行进一步操作。以下是每行代码的详细中文分析：
		if audio_adj != None:
			# 如果存在音频邻接矩阵（audio_adj）：
			if args.trans == 0:
				# 如果参数 args.trans 的值为0：
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			# 	计算音频特征：将音频嵌入（self.audio_embedding）与音频变换矩阵（self.audio_trans）相乘，
			# 	然后通过 LeakyReLU 激活函数处理得到音频特征。
			else:
				# 否则：
				audio_feats = self.audio_trans(self.audio_embedding)
		# 		对音频嵌入（self.audio_embedding）进行音频转换函数（self.audio_trans）操作得到音频特征。

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		# 将用户嵌入（self.uEmbeds）与归一化的图像特征（image_feats）连接起来，以生成图像的嵌入表示。E
		embedsImage = torch.spmm(image_adj, embedsImage)
		# 使用稀疏矩阵-矩阵乘法（Sparse Matrix Multiplication）操作，
		# 将图像邻接矩阵（image_adj）A与图像嵌入表示E（embedsImage）相乘，得到更新后的图像嵌入表示。Z

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		# 将用户嵌入（self.uEmbeds）与归一化的文本特征（text_feats）连接起来，以生成文本的嵌入表示。
		embedsText = torch.spmm(text_adj, embedsText)
		# 使用稀疏矩阵-矩阵乘法（Sparse Matrix Multiplication）操作，
		# 将文本邻接矩阵（text_adj）与文本嵌入表示（embedsText）相乘，得到更新后的文本嵌入表示。
		# 在这段代码中，除了处理音频特征外，它还将图像和文本特征与用户嵌入连接起来，并通过稀疏矩阵的乘法操作更新图像和文本的嵌入表示。
		# 这些操作通常用于将不同类型的特征结合起来以进行后续的模型训练或预测。

		if audio_adj != None:
			# 如果存在音频邻接矩阵（audio_adj）：
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			# 将用户嵌入（self.uEmbeds）与归一化的音频特征（audio_feats）连接起来，生成音频的嵌入表示。
			embedsAudio = torch.spmm(audio_adj, embedsAudio)
		# 	使用稀疏矩阵-矩阵乘法（Sparse Matrix Multiplication）操作，
		# 	将音频邻接矩阵（audio_adj）与音频嵌入表示（embedsAudio）相乘，得到更新后的音频嵌入表示。

		embeds1 = embedsImage
		# 把图像嵌入（embedsImage）赋值给embeds1。
		embedsLst1 = [embeds1]
		# 创建一个列表embedsLst1，并将embeds1添加到该列表中。
		for gcn in self.gcnLayers:
			# 对于self.gcnLayers中的每个GCN层：
			embeds1 = gcn(adj, embedsLst1[-1])#Zt
			# 使用当前GCN层（gcn）对邻接矩阵（adj）和前一层的输出（embedsLst1中的最后一个元素）进行操作，得到更新后的embeds1。
			embedsLst1.append(embeds1)
		# 	将更新后的embeds1添加到embedsLst1列表中。
		embeds1 = sum(embedsLst1)
		# 对embedsLst1列表中的所有元素求和，得到最终的embeds1表示。

		embeds2 = embedsText
		# 把文本嵌入（embedsText）赋值给embeds2。
		embedsLst2 = [embeds2]
		# 创建一个列表embedsLst2，并将embeds2添加到该列表中。
		for gcn in self.gcnLayers:
			# 对于self.gcnLayers中的每个GCN层：
			embeds2 = gcn(adj, embedsLst2[-1])
			# 使用当前GCN层（gcn）对邻接矩阵（adj）和前一层的输出（embedsLst2中的最后一个元素）进行操作，得到更新后的embeds2。
			embedsLst2.append(embeds2)
		# 	将更新后的embeds2添加到embedsLst2列表中。
		embeds2 = sum(embedsLst2)
		# 对embedsLst2列表中的所有元素求和，得到最终的embeds2表示。
		# 这段代码主要在处理嵌入表示的计算，包括将不同类型的嵌入信息与邻接矩阵结合，
		# 通过GCN（图卷积网络）层来更新和整合不同类型的嵌入表示，最终得到综合的图像和文本的嵌入表示。
		# 这些表示通常用于后续的图神经网络任务。

		if audio_adj != None:
			# 如果存在音频邻接矩阵（audio_adj）：
			embeds3 = embedsAudio
			# 将音频的嵌入表示（embedsAudio）赋值给embeds3。
			embedsLst3 = [embeds3]
			# 创建一个列表embedsLst3，并将embeds3添加到该列表中。
			for gcn in self.gcnLayers:
				# 对于self.gcnLayers中的每个GCN层：
				embeds3 = gcn(adj, embedsLst3[-1])
				# 使用当前GCN层（gcn）对邻接矩阵（adj）和前一层的输出（embedsLst3中的最后一个元素）进行操作，得到更新后的embeds3。
				embedsLst3.append(embeds3)
			# 	将更新后的embeds3添加到embedsLst3列表中。
			embeds3 = sum(embedsLst3)
		# 	对embedsLst3列表中的所有元素求和，得到最终的embeds3表示。

		if audio_adj == None:
			# 如果不存在音频邻接矩阵：
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		# 返回图像和文本嵌入表示中针对用户（args.user）进行分割后的结果。
		else:
			# 否则：
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]
		# 返回图像、文本和音频嵌入表示中针对用户（args.user）进行分割后的结果。


	def reg_loss(self):
		# 定义了一个名为 reg_loss 的方法，该方法属于特定的类（self指代类的实例）。
		ret = 0
		# 初始化一个变量ret，赋值为0，用于计算正则化损失的累加结果。
		ret += self.uEmbeds.norm(2).square()
		# 将用户嵌入（self.uEmbeds）的L2范数的平方加到ret变量中，用于计算用户嵌入的正则化损失。
		ret += self.iEmbeds.norm(2).square()
		# 将物品嵌入（self.iEmbeds）的L2范数的平方加到ret变量中，用于计算物品嵌入的正则化损失。
		return ret
# 	返回计算得到的正则化损失结果。
# 这段代码定义了一个正则化损失计算的方法reg_loss，
# 用于计算用户嵌入和物品嵌入的L2范数的平方之和作为正则化损失的值，该方法返回最终计算得到的正则化损失结果。

class GCNLayer(nn.Module):
	# 定义了一个名为GCNLayer的类，该类继承自nn.Module。
	def __init__(self):
		# 定义了GCNLayer类的初始化方法。
		super(GCNLayer, self).__init__()
	# 	调用父类nn.Module的初始化方法，确保正确初始化GCNLayer类的实例。

	def forward(self, adj, embeds):
		# 定义了GCNLayer类中的forward方法，用于执行前向传播操作，接收邻接矩阵（adj）和嵌入表示（embeds）作为输入。
		return torch.spmm(adj, embeds)
# 	在forward方法中，使用稀疏矩阵-矩阵乘法操作（torch.spmm），执行稀疏矩阵与密集矩阵的乘法运算
# 	将邻接矩阵（adj）与嵌入表示（embeds）进行相乘，得到GCN层的输出结果。
# 这段代码定义了一个GCN层类GCNLayer，其中包含一个前向传播方法forward，用于执行GCN层的操作，
# 即将邻接矩阵与嵌入表示相乘以得到更新后的嵌入表示。

class SpAdjDropEdge(nn.Module):   #边的随机丢弃操作，以一定比例保留边，同时调整边的权重，对图数据进行处理

	def __init__(self, keepRate):
		# 定义了SpAdjDropEdge类的初始化方法，接受一个参数keepRate表示保留边的比例。
		super(SpAdjDropEdge, self).__init__()
		# 调用父类nn.Module的初始化方法，确保正确初始化SpAdjDropEdge类的实例。
		self.keepRate = keepRate

	def forward(self, adj):
		# 定义了SpAdjDropEdge类中的forward方法，用于执行前向传播操作，接收邻接矩阵（adj）作为输入。
		vals = adj._values()
		# 提取邻接矩阵adj中的非零值并保存在vals中。
		idxs = adj._indices()
		# 提取邻接矩阵adj中的非零元素索引并保存在idxs中。
		edgeNum = vals.size()
		# 获取边的数量，即邻接矩阵中非零值的个数。
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)
		# torch.rand(edgeNum) 生成一个长度为 edgeNum 的随机张量，其值在 0 到 1 之间
		# + self.keepRate 将保留的比例 keepRate 加到每个随机值上，这样可以控制保留边的比例范围。
		# .floor() 取下限，将浮点数向下取整，确保掩码中的值最终只有 0 和 1
		# 根据边的数量生成一个掩码mask，用于随机丢弃边，保留一部分，其中根据keepRate控制保留的比例。

		newVals = vals[mask] / self.keepRate
		# 根据掩码mask选取对应位置的值，并根据keepRate调整边的权重，生成新的边权重。
		newIdxs = idxs[:, mask]
		# 根据掩码mask选取对应位置的索引，生成新的索引。

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
# 	根据新的边权重和索引，以及原始邻接矩阵adj的形状，创建一个新的稀疏矩阵并返回。
		
class Denoise(nn.Module):

	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		# 输入维度in_dims、输出维度out_dims、嵌入维度d_emb_size=10、是否进行归一化norm、dropout概率
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		# 将传入的输入维度in_dims保存在类的实例变量in_dims中。
		self.out_dims = out_dims
		# 将传入的输出维度out_dims保存在类的实例变量out_dims中。
		self.time_emb_dim = emb_size
		# 将传入的嵌入维度emb_size保存在类的实例变量time_emb_dim中。
		self.norm = norm
		# 将传入的norm参数保存在类的实例变量norm中，表示是否进行归一化。

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
		# 定义一个线性层emb_layer，用于嵌入维度的线性变换。

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		# 根据输入维度和嵌入维度计算出一个新的输入维度列表in_dims_temp，用于构建输入层。

		out_dims_temp = self.out_dims
		# 将输出维度保存在out_dims_temp中，用于构建输出层。

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		#  nn.ModuleList 类的实例，其中包含了多个线性层模块。
		# #使用列表推导式将输入层的线性层组成为一个ModuleList，用于处理输入数据的线性变换。
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
		# 使用列表推导式将输出层的线性层组成为一个ModuleList，用于处理输出数据的线性变换。

		self.drop = nn.Dropout(dropout)
		# 定义一个Dropout层，用于进行dropout操作，减少过拟合。
		self.init_weights()
	# 	调用init_weights方法来初始化模型的权重。
	# 这段代码定义了一个Denoise模块，包括了输入层、嵌入层、输出层以及对应的线性层和Dropout层等组件，用于执行去噪任务。

	def init_weights(self):
		# 定义了一个名为init_weights的函数，用于初始化模型中各个层的权重。
		for layer in self.in_layers:
			# 遍历模型中的输入层，准备初始化每个输入层的权重。
			size = layer.weight.size()
			# 获取当前层的权重张量的大小。
			std = np.sqrt(2.0 / (size[0] + size[1]))
			# 根据当前层的权重张量大小，计算用于初始化权重的标准差。
			layer.weight.data.normal_(0.0, std)
			# 使用均值为0、标准差为std的正态分布来初始化当前层的权重。
			layer.bias.data.normal_(0.0, 0.001)
		# 	使用均值为0、标准差为0.001的正态分布来初始化当前层的偏置项。
		
		for layer in self.out_layers:
			# 遍历模型中的输出层，准备初始化每个输出层的权重。
			size = layer.weight.size()
			# 获取当前输出层的权重张量的大小。
			std = np.sqrt(2.0 / (size[0] + size[1]))
			# 根据当前输出层的权重张量大小，计算用于初始化权重的标准差。
			layer.weight.data.normal_(0.0, std)
			# 使用均值为0、标准差为std的正态分布来初始化当前输出层的权重。
			layer.bias.data.normal_(0.0, 0.001)
		# 	使用均值为0、标准差为0.001的正态分布来初始化当前输出层的偏置项。

		size = self.emb_layer.weight.size()
		# 获取嵌入层的权重张量的大小。
		std = np.sqrt(2.0 / (size[0] + size[1]))
		# 根据嵌入层的权重张量大小，计算用于初始化权重的标准差。
		self.emb_layer.weight.data.normal_(0.0, std)
		# 使用均值为0、标准差为std的正态分布来初始化嵌入层的权重。
		self.emb_layer.bias.data.normal_(0.0, 0.001)
	# 	使用均值为0、标准差为0.001的正态分布来初始化嵌入层的偏置项。

	def forward(self, x, timesteps, mess_dropout=True):
		# 定义了一个前向传播方法forward，接受输入x、时间步timesteps 和 mess_dropout 参数。
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		# 计算出频率用于创建时间嵌入，通过指数计算出频率freqs。
		### math.log(10000)：计算以 10000 为底的对数。
		# torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32)：生成一个从 0 开始、最大值为 self.time_emb_dim//2 - 1 的浮点数张量
		#将上述生成的张量除以 (self.time_emb_dim//2)：这一步是为了将序列的取值范围缩放到 [0, 1] 区间
		# torch.exp(-math.log(10000) * 上一步结果)：对上一步得到的结果，分别取负的以 10000 为底的对数，然后对每个元素进行指数运算
		# ####长度为 self.time_emb_dim//2 的一维张量
		temp = timesteps[:, None].float() * freqs[None]
		# 计算临时变量temp，将时间步与频率相乘。temp 张量将具有形状为 (N, M)，其中 N 和 M 分别是 timesteps 和 freqs 张量的长度
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		# 根据临时变量temp计算时间嵌入time_emb，包括余弦和正弦项。time_emb 张量的维度将是 (N, 2*M)
		if self.time_emb_dim % 2:
			# 检查时间嵌入维度是否为奇数。
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		# 	如果时间嵌入维度是奇数，则在时间嵌入中添加一个零项。
		emb = self.emb_layer(time_emb)
		# 将时间嵌入通过嵌入层emb_layer进行线性变换得到嵌入项emb。
		if self.norm:
			# 如果设置了norm标志，则执行归一化操作。
			x = F.normalize(x)
		# 	对输入x进行归一化操作。
		if mess_dropout:
			# 如果启用了mess_dropout，则执行Dropout操作。
			x = self.drop(x)
		# 	对输入x应用Dropout。
		h = torch.cat([x, emb], dim=-1)
		# 将归一化后的输入x和嵌入项emb按最后一个维度连接起来得到h。
		for i, layer in enumerate(self.in_layers):
			# 遍历输入层的线性层，准备执行前向传播操作。
			h = layer(h)
			# 对输入h应用当前线性层layer的变换。
			h = torch.tanh(h)
		# 	使用双曲正切激活函数对结果h进行非线性激活。
		for i, layer in enumerate(self.out_layers):
			# 遍历输出层的线性层，准备执行前向传播操作。
			h = layer(h)
			# 对输入h应用当前输出层layer的变换。
			if i != len(self.out_layers) - 1:
				# 如果不是最后一层，则执行下面操作。
				h = torch.tanh(h)
		# 		使用双曲正切激活函数对结果h进行非线性激活。

		return h
# 	返回最终的输出h。

class GaussianDiffusion(nn.Module):
	# 定义了一个名为GaussianDiffusion的类，它继承自nn.Module。
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		# 定义了类的初始化方法，接收噪声规模，最小噪声，最大噪声，步数和beta_fixed参数。
		super(GaussianDiffusion, self).__init__()
		# 调用父类nn.Module的初始化方法。

		self.noise_scale = noise_scale
		# 将输入的噪声规模存储到实例变量noise_scale中。
		self.noise_min = noise_min
		# 将输入的最小噪声存储到实例变量noise_min中。
		self.noise_max = noise_max
		# 将输入的最大噪声存储到实例变量noise_max中。
		self.steps = steps
		# 将输入的步数存储到实例变量steps中。

		if noise_scale != 0:
			# 如果噪声规模不为零，则执行以下操作。
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			# 调用get_betas方法得到beta值，并将其转换为torch.tensor，数据类型为torch.float64，并移到GPU上。
			if beta_fixed:
				# 如果beta_fixed为真，则执行以下操作。
				self.betas[0] = 0.0001
			# 	将第一个beta值设为0.0001。

			self.calculate_for_diffusion()
	# 		调用calculate_for_diffusion方法进行扩散计算。
	# 这段代码定义了GaussianDiffusion类，它根据输入的参数进行初始化并计算扩散参数。如果噪声规模不为零，它会计算beta值并执行相应操作。

	def get_betas(self):
		# 定义了一个get_betas方法，该方法属于类中的实例方法。
		start = self.noise_scale * self.noise_min
		# 计算起始值，为噪声规模乘以最小噪声。
		end = self.noise_scale * self.noise_max
		# 计算结束值，为噪声规模乘以最大噪声。
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		# 生成一个包含self.steps个数值的等差数组，范围从start到end，数据类型为np.float64，存储在variance中。
		alpha_bar = 1 - variance
		# 计算alpha_bar，即1 - variance。
		betas = []
		# 初始化一个空列表用于存储beta值。
		betas.append(1 - alpha_bar[0])
		# 将第一个beta值加入到betas列表中，计算方式为1 - alpha_bar[0]。
		for i in range(1, self.steps):
			# 遍历从1到self.steps - 1的范围。
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		# 	将最小值取为1 - alpha_bar[i] / alpha_bar[i-1]与0.999的最小值，并将其添加到betas列表中。
		return np.array(betas)
	# 将beta值列表转换为NumPy数组并返回。
	# 这段代码定义了一个get_betas方法，用于计算一系列的beta值，最终返回这些值的数组。
	# 通过计算起始值、结束值以及相应的数学计算，生成了一组beta值并将其返回。

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		# 计算alphas，即1.0 - self.betas。
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		# 计算alphas数组的累积乘积，并将结果移到GPU上，存储在self.alphas_cumprod中。
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		# 构建self.alphas_cumprod的前一个值数组并在开头添加一个值为1.0的元素，存储在self.alphas_cumprod_prev中。
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()
		# 构建self.alphas_cumprod的后一个值数组并在末尾添加一个值为0.0的元素，存储在self.alphas_cumprod_next中。

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		# 计算self.alphas_cumprod的平方根，并存储在self.sqrt_alphas_cumprod中。
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		# 计算1.0 - self.alphas_cumprod的平方根，并存储在self.sqrt_one_minus_alphas_cumprod中。
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		# 计算1.0 - self.alphas_cumprod的对数，并存储在self.log_one_minus_alphas_cumprod中。
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		# 计算1.0 / self.alphas_cumprod的平方根，并存储在self.sqrt_recip_alphas_cumprod中。
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
		# 计算1.0 / self.alphas_cumprod - 1的平方根，并存储在self.sqrt_recipm1_alphas_cumprod中。

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		# 计算后验方差，并存储在self.posterior_variance中。
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		# 对后验方差进行对数计算并剪辑，存储在self.posterior_log_variance_clipped中。
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		# 计算后验均值系数1，并存储在self.posterior_mean_coef1中。
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))
	# 	计算后验均值系数2，并存储在self.posterior_mean_coef2中。
	# 这段代码中包含了对各种变量和参数进行计算与处理的步骤，用于后续的扩散计算和相关操作。

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		# denoise_model_feature, batch_item, args.sampling_steps=0, args.sampling_noise=False
		if steps == 0:
			x_t = x_start
		# 如果步数为0，则将当前值设置为初始值x_start。
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			# 否则，创建一个包含tensors的Tensor，维度与x_start相同，值为steps-1，并将其转移到GPU上。
			x_t = self.q_sample(x_start, t)
		# 	调用q_sample方法获得下一个时间步的值x_t。
		
		indices = list(range(self.steps))[::-1]
		# 生成一个倒序的时间步骤索引列表。

		for i in indices:
			# 对时间步骤索引进行遍历。
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			# 创建包含当前步数索引的Tensor，并将其转移到GPU上。
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			# 获取给定模型、x_t和时间步数t的模型均值和对数方差。denoise_model_image
			if sampling_noise:
				# 如果需要进行噪声采样：
				noise = torch.randn_like(x_t)
				# 生成与x_t相同形状的随机噪声。
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				# 创建一个非零元素掩码，用于处理噪声采样的情况。
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			# 	通过模型均值、对数方差和噪声生成下一个时间步的值x_t。
			else:
				# 如果不需要进行采样噪声：
				x_t = model_mean
		# 		直接将x_t设置为模型均值。
		return x_t
	# 返回最终的时间步结果x_t。
	# 这段代码实现了基于给定模型的采样过程，根据模型的均值和方差计算出下一个时间步的值。
	# 根据参数sampling_noise的设定，可能会包含噪声采样的过程。

	def q_sample(self, x_start, t, noise=None):
		# 定义一个函数q_sample，接受初始值x_start、时间步t以及可选的噪声noise作为参数。
		if noise is None:
			noise = torch.randn_like(x_start)
		# 	如果没有传入噪声，生成一个与x_start相同形状的随机噪声。
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	# 返回使用sqrt_alphas_cumprod和sqrt_one_minus_alphas_cumprod对应的值，结合初始值x_start和噪声，生成下一个时间步的采样值。

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		# 定义了一个私有方法_extract_into_tensor，接受数组arr、时间步长序列timesteps和广播形状broadcast_shape作为参数。
		arr = arr.cuda()
		# 将输入的数组arr移到GPU上。
		res = arr[timesteps].float()
		# 获取数组arr中与时间步timesteps对应位置的数值，转换为浮点数类型，并存储在res中。
		while len(res.shape) < len(broadcast_shape):
			# 当res的维度少于广播形状的维度时，执行循环。
			res = res[..., None]
		# 	为res添加一个维度。
		return res.expand(broadcast_shape)
	# 将res扩展到与广播形状相匹配的维度并返回。
	# 这段代码实现了从给定的数组arr中提取特定时间步对应的值，用于计算下一个时间步的采样值。
	# q_sample函数利用提取的值生成下一个时间步的采样结果。

	def p_mean_variance(self, model, x, t):
		# denoise_model, x_t, t
		# 定义了一个函数p_mean_variance，接受模型model、输入值x和时间步t作为参数。
		model_output = model(x, t, False)
		# 通过调用模型model，传入输入值x和时间步t，得到模型的输出结果model_output。

		model_variance = self.posterior_variance
		# 将self.posterior_variance赋值给model_variance，后者表示后验方差。
		model_log_variance = self.posterior_log_variance_clipped
		# 将修剪后的后验对数方差self.posterior_log_variance_clipped赋值给model_log_variance。

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		# 使用私有方法_extract_into_tensor从model_variance中提取特定时间步的值，以匹配输入值x的形状，并赋给model_variance。
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
		# 同样使用_extract_into_tensor方法，从model_log_variance中提取特定时间步的对数方差值，与输入值x的形状相匹配，并赋给model_log_variance。

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		# 计算模型的均值，利用系数self.posterior_mean_coef1和self.posterior_mean_coef2与模型输出model_output和输入值x的线性组合。
		
		return model_mean, model_log_variance
	# 返回计算得到的模型均值model_mean和模型对数方差model_log_variance。
	# 这段代码实现了根据模型输出和先验方差、对数方差计算模型的均值和方差。
	# 通过给定的系数和相关值，计算出模型在特定时间步的均值和方差，以用于后续的采样过程。

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		#denoise_model_feature, batch_item, iEmbeds, batch_index, all_feats
		batch_size = x_start.size(0)
		# 获取输入值x_start的批量大小。  1024

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		# 生成一个在0到self.steps之间随机整数的 Tensor，形状与批量大小相同，并将其移动到GPU上。
		noise = torch.randn_like(x_start)
		# 生成与x_start相同形状的随机噪声。
		if self.noise_scale != 0:
			# 如果噪声缩放参数不为0：
			x_t = self.q_sample(x_start, ts, noise)   #添加噪声x_start + noise
		# 	通过调用q_sample方法生成下一个时间步的值x_t，考虑了噪声。
		else:
			x_t = x_start
		# 	否则，直接将x_t设置为x_start。

		model_output = model(x_t, ts) #调用Denoise的forward方法  正向扩散加噪
		# 通过模型model计算给定时间步和值的模型输出结果model_output。

		mse = self.mean_flat((x_start - model_output) ** 2)
		# 计算均方误差（MSE）作为损失函数，衡量输入值和模型输出结果之间的差异。

		weight = self.SNR(ts - 1) - self.SNR(ts)
		# 计算权重，使用信噪比（SNR）对当前时间步和上一个时间步进行计算。
		weight = torch.where((ts == 0), 1.0, weight)
		# 在ts等于0时，将权重设置为1.0。

		diff_loss = weight * mse
		# 利用权重调整MSE得到差异损失diff_loss。Elbo

		usr_model_embeds = torch.mm(model_output, model_feats)
		#                            预测u-i交互     模态特征
		# 计算用户模型嵌入，对模型输出结果和模型特征进行矩阵乘法操作。
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		#                      原始u-i交互   物品id嵌入
		# 计算用户ID嵌入，对输入值和物品嵌入进行矩阵乘法运算。

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
		# 计算嵌入一致性损失，衡量用户模型嵌入与用户ID嵌入之间的差异。msi

		return diff_loss, gc_loss
	# 返回差异损失diff_loss和嵌入一致性损失gc_loss作为训练损失。
		
	def mean_flat(self, tensor):
		# 定义了一个函数mean_flat，该函数用于计算张量tensor沿着除第一维外其他维度的均值。
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	# 返回计算得到的张量tensor在除第一维外其他维度上的均值。
	# 通过指定维度参数dim为除第一维外的所有维度，实现对这些维度上的均值计算。
	
	def SNR(self, t):
		# 定义了一个函数SNR, 该函数用于计算信噪比（Signal-to-Noise Ratio）。
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		# 将self.alphas_cumprod移到GPU上进行计算。
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
# 	返回在时间步t处的self.alphas_cumprod除以1 - self.alphas_cumprod[t]计算得到的信噪比值。
# 这段代码实现了计算张量的均值以及计算信噪比的功能。
# mean_flat函数用于计算张量在指定维度上的均值，而SNR函数用于根据给定的时间步计算信噪比，
# 其中涉及对self.alphas_cumprod的GPU计算和数学运算。
