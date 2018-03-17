import numpy as np
import numpy.linalg as la


N = 5 #Number of basis vectors
D = 3 #dimension of the basis vectors
A = np.array([[1,2,3,4,5],[0,1,0,0,0],[2,0,3,0,1]])
y = np.array([1,0.2,2])
gamma = 0.5

# Initialization
x = np.zeros([N,])
theta = np.zeros([N,])
active_set = set([])


while(True):
    cond_b = True
    diff = -2*np.matmul(A.T, y-np.matmul(A,x));
    zero_coff = np.where(x==0)[0]
    i = np.argmax(np.abs(diff[zero_coff]))
    
    # Activate xi if improves the objective function locally
    if diff[i] > gamma :
        theta[i] = -1
        active_set.add(i)
    if diff[i] < -gamma :
        theta[i] = 1
        active_set.add(i)
    
    while(True):
        cond_a = True
        # Feature sign step
        ind = np.array(list(active_set))
        A_hat = A[:,ind]
        x_hat = x[ind]
        theta_hat = theta[ind]
        
        x_hat_new = np.matmul(la.inv(np.matmul(A_hat.T, A_hat)),np.matmul(A_hat.T,y)-gamma*0.5*theta_hat)
        print(x_hat_new)
        # Discrete line search
        ls_dir = x_hat_new - x_hat
        step = np.arange(0,1,0.01)
        min_val = 100000
        for k in range(len(step)):
            x_chk = x_hat + step[k]*ls_dir
            obj_val = la.norm(y-np.matmul(A_hat,x_chk))**2+gamma*np.dot(theta_hat,x_chk)
            if obj_val<min_val:
                min_val = obj_val
                x_hat_min = x_chk
        
        x_hat = x_hat_min
        x[np.array(list(active_set))] = x_hat
        
        # remove zero coefficients of x_hat from the active set 
        z_ind = np.where(np.abs(x_hat)<1e-3)[0]
        for kk in range(len(z_ind)):
            print('chk chk \n')
            active_set.remove(ind[z_ind[kk]])
        
        theta = np.sign(x)
        
        #Check for optimality condition (a)
        diff = -2*np.matmul(A.T, y-np.matmul(A,x))
        nzero_xi = np.where(x!=0)[0]
        for kk in range(len(nzero_xi)):
            if np.abs(diff[nzero_xi[kk]]+gamma*np.sign(x[nzero_xi[kk]])) > 1e-6:
                cond_a = False # Condition A not satisfied
        
        if cond_a==True: # Condition A satisfied
            break

    print("Checking Condition B")
    #Check for optimality condition (b)
    diff = -2*np.matmul(A.T, y-np.matmul(A,x))
    zero_xi = np.where(np.abs(x)==0)[0]
    for kk in range(len(zero_xi)):
        if np.abs(diff[zero_xi[kk]]) > gamma:
            cond_b = False # Condition B not satisfied
    
    if cond_b==True: # Condition B satisfied
        break
print(x)