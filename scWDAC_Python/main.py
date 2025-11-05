import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'.\datasets', nargout=0)

## load dataset
data = eng.load(r'.\datasets\data1_sim.mat')
X = data['X']
true_label = data['true_label']

result = eng.scWDAC(X, true_label)
print(result)

eng.quit()

