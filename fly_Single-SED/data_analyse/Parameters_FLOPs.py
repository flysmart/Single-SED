import torch
import models.densenet
import models.resnet
import models.inception
import models.bilstm
import models.cnn
import models.crnn
import models.crnn1
import models.transformer
from models.conformer import model
from ptflops import get_model_complexity_info


# 统计CNN模型的参数量和计算复杂度
model_cnn = models.cnn.CNN(3, 50)
device = torch.device('cpu')
model_cnn.to(device)
flops_cnn, params_cnn = get_model_complexity_info(model_cnn, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('CNN模型参数量：' + params_cnn)
print('CNN模型计算复杂度：' + flops_cnn)

# 统计DenseNet模型的参数量和计算复杂度
model_densenet = models.densenet.DenseNet("ESC")
model_densenet.to(device)
flops_densenet, params_densenet = get_model_complexity_info(model_densenet, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('DenseNet模型参数量：' + params_densenet)
print('DenseNet模型计算复杂度：' + flops_densenet)

# 统计CRNN（DenseNet + BiLSTM）模型的参数量和计算复杂度
model_crnn_densenet_bilstm = models.crnn.CRNN(512, 2, 50)
model_crnn_densenet_bilstm.to(device)
flops_crnn_densenet_bilstm, params_crnn_densenet_bilstm = get_model_complexity_info(model_crnn_densenet_bilstm, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('CRNN（DenseNet + BiLSTM）模型参数量：' + params_crnn_densenet_bilstm)
print('CRNN（DenseNet + BiLSTM）模型计算复杂度：' + flops_crnn_densenet_bilstm)

# 统计Conformer模型的参数量和计算复杂度
model_conformer = model.AudioConformer()
model_conformer.to(device)
flops_conformer, params_conformer = get_model_complexity_info(model_conformer, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('Conformer模型参数量：' + params_conformer)
print('Conformer模型计算复杂度：' + flops_conformer)

# 统计ResNet模型的参数量和计算复杂度
model_resnet = models.resnet.ResNet("ESC")
model_resnet.to(device)
flops_resnet, params_resnet = get_model_complexity_info(model_resnet, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('ResNet模型参数量：' + params_resnet)
print('ResNet模型计算复杂度：' + flops_resnet)

# 统计Inception模型的参数量和计算复杂度
model_inception = models.inception.Inception("ESC")
model_inception.to(device)
flops_inception, params_inception = get_model_complexity_info(model_inception, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('Inception模型参数量：' + params_inception)
print('Inception模型计算复杂度：' + flops_inception)

# 统计Transformer模型的参数量和计算复杂度
model_transformer = models.transformer.AudioTransformer(384, 512, 6, 50)
model_transformer.to(device)
flops_transformer, params_transformer = get_model_complexity_info(model_transformer, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('Transformer模型参数量：' + params_transformer)
print('Transformer模型计算复杂度：' + flops_transformer)

# 统计CRNN（CNN + BiLSTM）模型的参数量和计算复杂度
model_crnn_cnn_bilstm = models.crnn1.CRNN(3, 512, 2, 50)
model_crnn_cnn_bilstm.to(device)
flops_crnn_cnn_bilstm, params_crnn_cnn_bilstm = get_model_complexity_info(model_crnn_cnn_bilstm, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('CRNN（CNN + BiLSTM）模型参数量：' + params_crnn_cnn_bilstm)
print('CRNN（CNN + BiLSTM）模型计算复杂度：' + flops_crnn_cnn_bilstm)

# 统计BiLSTM模型的参数量和计算复杂度
model_bilstm = models.bilstm.BiLSTM(384, 512, 2, 50)
model_bilstm.to(device)
flops_bilstm, params_bilstm = get_model_complexity_info(model_bilstm, (3, 128, 250), as_strings=True, print_per_layer_stat=False)
print('BiLSTM模型参数量：' + params_bilstm)
print('BiLSTM模型计算复杂度：' + flops_bilstm)


