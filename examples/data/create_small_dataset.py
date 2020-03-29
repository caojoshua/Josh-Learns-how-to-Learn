
NUM_LINES = 1000

def write_small(in_name, out_name):
	f = open(in_name, "r")
	out = open(out_name, "w")
	for line in f.readlines()[0:NUM_LINES]:
		out.write(line)
	
write_small("train.csv", "train_small.csv")
write_small("test.csv", "test_small.csv")