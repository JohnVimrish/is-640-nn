from nn import MLP as m

x = [2.0, 3.0,-1.0]
n = m(3,[4,4,1])
print(n(x))

xs = [[2.0,3.0,-1.0],[3.0, -1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]]
ys  = [1.0,-1.0,-1.0,1.0]

for k in range (20) :
    ypredictability = [n(x) for x in xs ]
    loss =   sum([(yout-yin)**2 for  yout ,yin in zip(ys,ypredictability)])

    for p in n.parameters() :
        p.grad = 0.0
    loss.backward()


    for p in n.parameters() :
        p.data =+ 0.1 *  p.grad
    
    print(k,loss.data)


print (ypredictability)