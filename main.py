from nn import MLP as m

x = [2.0, 3.8,-19.3]
n = m(3,[4,4,1])
print(n(x))

xs = [[3.0,-1.0,2.5],[2.0, 0.8,-1.3],[-1.5,0.5,2.3],[3.5,6.9,-0.3]]
ys  = [1.0,-4.0,-1.0,1.0]

for k in range (5) :
    ypredictability = [n(x) for x in xs ]
    loss =   sum([(yout-yin)**2 for  yout ,yin in zip(ys,ypredictability)])

    for p in n.parameters() :
        p.grad = 0.0
    loss.backward()


    for p in n.parameters() :
        p.data =+ 0.1 *  p.grad
    
    print(k,loss.data)


print (ypredictability)