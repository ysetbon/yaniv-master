import numpy as np, pickle
from training.yaniv.contest_fable_lab import equity
T=np.load('results/contest-fable-logs/wtable.npy')
oc=pickle.load(open('results/contest-fable-logs/wtable.outcomes.pkl','rb'))
oc=np.array(oc)
print(len(oc), 'rounds; mean adds', oc.mean(0))
# winner's gain distribution: loser's add when the other adds 0 and add<30ish
pooled=np.concatenate([oc,oc[:,::-1]])
win=pooled[(pooled[:,0]==0)&(pooled[:,1]>0)][:,1]   # opp adds b
lose=pooled[(pooled[:,0]>0)&(pooled[:,1]==0)][:,0]
print('P(win round)',len(win)/len(pooled),'mean opp add when I win',win.mean(),'pctl',np.percentile(win,[10,25,50,75,90]))
print('share of losses >=30 (caught-like):',(lose>=30).mean())
E=lambda s,t: equity(T,s,t)
bands=[0,10,20,30,40,45,55,60,65,70,75,80,85,90,95,99]
print('\n(a) value of 1 point for me (x100 pp win chance): avg over 5-pt step, -(E(s+5,t)-E(s,t))/5 using raw table (no cut inside)')
print('s\\t  '+' '.join(f'{t:5d}' for t in bands))
for s in bands:
    row=[]
    for t in bands:
        s2=min(s+5,99); 
        d=(T[s,t]-T[s2,t])/max(1,s2-s)*100
        row.append(f'{d:5.2f}')
    print(f'{s:3d}  '+' '.join(row))
print('\n value of 1 point added to OPP (pp):')
for s in bands:
    row=[]
    for t in bands:
        t2=min(t+5,99)
        d=(T[s,t2]-T[s,t])/max(1,t2-t)*100
        row.append(f'{d:5.2f}')
    print(f'{s:3d}  '+' '.join(row))
print('\nE table (pp)')
for s in bands:
    print(f'{s:3d}  '+' '.join(f'{T[s,t]*100:5.1f}' for t in bands))
# (b) call: gain of successful call = E_b eq(s,t+b)-eq(s,t), b ~ win distribution; cost of caught with p=5: eq(s,t)-eq(s+35,t)
def gain(s,t): return np.mean([E(s,t+b) for b in win])-E(s,t)
def cost(s,t,p=5): return E(s,t)-E(s+p+30,t)
r0=gain(0,0)/cost(0,0)
print('\n(b) gain(pp)/cost(pp)/ratio rel 0-0 ; r0=',r0, gain(0,0),cost(0,0))
for s in bands:
    row=[]
    for t in bands:
        g,c=gain(s,t),cost(s,t)
        row.append(f'{g*100:4.0f}/{c*100:4.0f}/{(g/c/r0 if c>1e-9 else 99):4.1f}')
    print(f'{s:3d}  '+' '.join(row))
