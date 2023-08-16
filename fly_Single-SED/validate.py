import torch
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
def evaluate(model, device, test_loader):
	correct = 0
	total = 0
	model.eval()
	targets = []
	predictions = []
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0].to(device)
			target = data[1].squeeze(1).to(device)
			targets.extend(target.tolist())
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			predictions.extend(predicted.tolist())
			total += target.size(0)
			correct += (predicted == target).sum().item()
			f1 = f1_score(targets, predictions, average='macro')
			cm = confusion_matrix(targets, predictions)
			cm = pd.DataFrame(cm)

	return (100*correct/total), f1, cm
