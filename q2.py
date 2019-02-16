import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

class q2(object):
	def __init__(self):
		self.cancer_data = pd.read_csv("./ovariancancer.csv", header=None).values
		self.cancer_labels = pd.read_csv("./ovariancancer_labels.csv", header=None).values

		self.iter_counts = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
		self.learning_rates = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

		self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

		self.weights, self.w, self.forward_weights, self.backward_weights, self.confusion_matrix = [], [], [], [], []

		self.feature_set_forward, self.feature_set_backward = [], []

		self.parameters = ()

	def divide(self):
		one_start = np.where(self.cancer_labels == 1)[0][0]	
		zero_start = np.where(self.cancer_labels == 0)[0][0]

		self.X_train, self.X_test, self.y_train, self.y_test = np.concatenate([self.cancer_data[one_start + 20: zero_start, :], self.cancer_data[zero_start + 20:, :]]), \
		np.concatenate([self.cancer_data[:20, :], self.cancer_data[zero_start: zero_start + 20, :]]), \
		np.concatenate([self.cancer_labels[one_start + 20: zero_start], self.cancer_labels[zero_start + 20:]]), \
		np.concatenate([self.cancer_labels[:20], self.cancer_labels[zero_start: zero_start + 20]])

	def k_fold(self, X_train, y_train, rate, count, div, k = 0.5): 
		overall_accuracy = 0
		for x in xrange(5):
			start = x * div
			end = x * div + div
			X_valid = X_train[start : end, :]
			y_valid = y_train[start : end]
			X_inner_train = np.concatenate([X_train[:start, :], X_train[end:, :]])
			y_inner_train = np.concatenate([y_train[:start], y_train[end:]])
			self.w = np.zeros((X_inner_train.shape[1], 1))

			#w0 = np.ones((X_inner_train.shape[0], 1))
			#np.concatenate([w0, X_inner_train], axis=1)
			##Training
			for x in xrange(count):
				z = np.dot(X_inner_train, self.w)
				probs = 1 / (1 + np.exp(-z))
				gradient = np.dot(X_inner_train.T, y_inner_train - probs)
				self.w += rate * gradient
			##Validate
			predictions = []
			z = np.dot(X_valid, self.w)
			probs = 1 / (1 + np.exp(-z))
			for prob in probs:
				predictions.append(1) if prob > k else predictions.append(0)
			current_accuracy = self.compare(predictions, y_valid)
			overall_accuracy += current_accuracy
		return overall_accuracy

	def compare(self, predictions, y, test = False, curve = False):
		tp, tn, fp, fn = 0, 0, 0, 0
		for x in xrange(len(y)):
			if predictions[x] == 1 and y[x] == 1:
				tp += 1
			elif predictions[x] == 0 and y[x] == 0:
				tn += 1
			elif predictions[x] == 1 and y[x] == 0:
				fp += 1
			else:
				fn += 1

		if curve:
			self.confusion_matrix = [tp, tn, fp, fn]

		if test:
			print "True Positives: {}".format(tp)
			print "True Negatives: {}".format(tn)
			print "False Positives: {}".format(fp)
			print "False Negatives: {}".format(fn)

		return float((tp + tn)) / float((tp + tn + fp + fn))

	def test(self, X_test, y_test, weights, k = 0.5, curve = False):
		predictions = []
		z = np.dot(X_test, weights)
		probs = 1 / (1 + np.exp(-z))
		for prob in probs:
			predictions.append(1) if prob > k else predictions.append(0)
		current_accuracy = self.compare(predictions, y_test, True, curve)

	def graph(self, x, y, title, x_label, y_label):
		area = auc(x, y)
		plt.figure()
		plt.plot(x, y, color='darkorange', lw=1, label='Area = %0.2f' % area)
		plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title)
		plt.legend(loc = "lower right")
		plt.show()

	def forward_selection(self, features_added):
		### Train Forward Selection ###
		print "Doing forward selection now"
		rate, count = self.parameters[0], self.parameters[1]
		max_accuracy = 0
		max_id = -1
		div = len(self.X_train) / 5
		for x in xrange(features_added):
			max_k_fold_accuracy = 0
			for x in xrange(len(self.X_train[0])):
				if x not in self.feature_set_forward:
					self.feature_set_forward.append(x)
					k_fold_accuracy = self.k_fold(self.X_train[:, self.feature_set_forward], self.y_train, rate, count, div) / 5.0
					if k_fold_accuracy > max_k_fold_accuracy:
						max_k_fold_accuracy = k_fold_accuracy
						max_id = x
						self.forward_weights = self.w
					self.feature_set_forward.pop()
			if max_id not in self.feature_set_forward:
				print "Adding: {}".format(max_id)
				self.feature_set_forward.append(max_id)
		print self.feature_set_forward
		print self.forward_weights

		#### Test with Forward Selection ####
		print "Confusion Matrix for Forward Selection"
		self.test(self.X_test[:, self.feature_set_forward], self.y_test, self.forward_weights)

	def backward_selection(self, features_deleted):
		### Train Backward Selection ###
		print "Doing backward selection now"
		self.feature_set_backward = [x for x in xrange(len(self.X_train[0]))]
		rate, count = self.parameters[0], self.parameters[1]
		max_accuracy = 0
		max_id = -1
		div = len(self.X_train) / 5
		for x in xrange(features_deleted):
			max_k_fold_accuracy = 0
			for x in xrange(len(self.X_train[0])):
				if x in self.feature_set_backward:
					self.feature_set_backward.remove(x)
					k_fold_accuracy = self.k_fold(self.X_train[:, self.feature_set_backward], self.y_train, rate, count, div) / 5.0
					if k_fold_accuracy > max_k_fold_accuracy:
						max_k_fold_accuracy = k_fold_accuracy
						max_id = x
						self.backward_weights = self.w
					self.feature_set_backward.append(x)
			if max_id in self.feature_set_backward:
				print "Removing: {}".format(max_id)
				self.feature_set_backward.remove(max_id)
		print self.feature_set_backward
		print self.backward_weights

		#### Test with Backward Selection ###
		print "Confusion Matrix for Backward Selection"
		self.test(self.X_test[:, self.feature_set_backward], self.y_test, self.backward_weights)

	def q2_1(self):
		self.divide()

		div = len(self.X_train) / 5
		max_accuracy = 0
		for rate in self.learning_rates:
			for count in self.iter_counts:
				k_fold_accuracy = self.k_fold(self.X_train, self.y_train, rate, count, div) / 5.0
				if k_fold_accuracy > max_accuracy:
					self.parameters = (rate, count)
					self.weights = self.w
					max_accuracy = k_fold_accuracy
				
		
		print "Best learning rate: {}".format(self.parameters[0])
		print "Best iteration count: {}".format(self.parameters[1])
		
		print "Confusion Matrix for Logistic Regression"
		self.test(self.X_test, self.y_test, self.weights)

	def q2_2(self):
		self.forward_selection(5)
		self.backward_selection(5)
		print "Are both the forward and backward selction models equal?"
		print set(self.feature_set_forward) == set(self.feature_set_backward)

	def q2_3(self):
		k_div = 1.0 / len(self.y_test)
		#### Forward Selection ####
		fpr = []
		tpr = []
		precision = []
		k = 0
		while k < 1:
			self.test(self.X_test[:, self.feature_set_forward], self.y_test, self.forward_weights, k, curve = True)
			f = float(self.confusion_matrix[2]) / float((self.confusion_matrix[2] + self.confusion_matrix[1])) if (self.confusion_matrix[2] + self.confusion_matrix[1]) != 0 else 0
			fpr.append(f)
			t =  float(self.confusion_matrix[0]) / float((self.confusion_matrix[0] + self.confusion_matrix[3])) if (self.confusion_matrix[0] + self.confusion_matrix[3]) != 0 else 0
			tpr.append(t)
			p = float(self.confusion_matrix[0]) / float((self.confusion_matrix[0] + self.confusion_matrix[2])) if (self.confusion_matrix[0] + self.confusion_matrix[2]) != 0 else 0
			precision.append(p)
			k += k_div
		#### Plotting graphs for Forward Selection ####-
		self.graph(fpr, tpr, 'Forward Selection ROC Curve', 'False Positive Rate', 'True Positive Rate')
		self.graph(tpr, precision, 'Forward Selection Precision-Recall Curve', 'Recall', 'Precision')
		
		#### Backward Selection ####
		fpr = []
		tpr = []
		precision = []
		k = 0
		while k < 1:
			print len(self.feature_set_backward)
			self.test(self.X_test[:, self.feature_set_backward], self.y_test, self.backward_weights, k, curve = True)
			print self.confusion_matrix
			f = float(self.confusion_matrix[2]) / float((self.confusion_matrix[2] + self.confusion_matrix[1])) if (self.confusion_matrix[2] + self.confusion_matrix[1]) != 0 else 0
			fpr.append(f)
			t =  float(self.confusion_matrix[0]) / float((self.confusion_matrix[0] + self.confusion_matrix[3])) if (self.confusion_matrix[0] + self.confusion_matrix[3]) != 0 else 0
			tpr.append(t)
			p = float(self.confusion_matrix[0]) / float((self.confusion_matrix[0] + self.confusion_matrix[2])) if (self.confusion_matrix[0] + self.confusion_matrix[2]) != 0 else 0
			precision.append(p)
			k += k_div

		#### Plotting graphs for Backward Selection ####
		print fpr
		print tpr
		print precision
		self.graph(fpr, tpr, 'Backward Selection ROC Curve', 'False Positive Rate', 'True Positive Rate')
		self.graph(tpr, precision, 'Backward Selection Precision-Recall Curve', 'Recall', 'Precision')
		

q2 = q2()
q2.q2_1()
q2.q2_2()
q2.q2_3()
