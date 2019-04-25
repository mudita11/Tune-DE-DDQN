# Tune-DE-DDQN

training_set is generated using following commands:
x = np.random.choice(2160, 1080,replace = False)
with open('training_set','w+') as f:
    for i in x:
        f.write(str(i)+'\n')
