l = []
size = 7590
ks = 512
for n in range(0,40):
    for s in range(50,100):
        p = (ks-s)*n - (size-ks)
        if int(p) == p and p>0:
            l.append([p,s,n])
            
print(l)