"""
MATH3036 CW2 module

@author: Thea Hellen
"""

import numpy as np
import matplotlib.pyplot as plt



def SDmethod(A,b,x0,kmax):
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    
    # Store initial approximation
    x = x0
    x_array[:,[0]] = x
    
    # Initial r
    r = b - A @ x0
         
    # SD loop
    for k in np.arange(kmax):
        # Step length
        a = (r.T @ r) / (r.T @ A @ r)
        
        # Update approximation 
        x =  x + a * r
        
        # Update residual
        r = b - A @ x
        
        # Store
        x_array[:,[k+1]] = x

    # Return
    return x_array



def CGmethod(A,b,x0,kmax):
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    
    # Store initial approximation
    x = x0
    x_array[:,[0]] = x
    
    # Initial r and p
    r_old = b - A @ x0
    p = r_old
    
    # CG loop
    for k in np.arange(kmax):

        # Step length
        if np.linalg.norm(p) < 1e-15:
            a = 0.0
        else:
            a = (r_old.T @ r_old) / (p.T @ A @ p)
            
        # Update approximation 
        x =  x + a * p
        
        # Update residual 
        r = r_old - a * A @ p
        
        # Update search direction
        if np.linalg.norm(r_old) < 1e-15:
            b = 0.0
        else:
            b = r.T @ r / (r_old.T @ r_old)
        p = r + b*p
        
        # Update r_old
        r_old = r
        
        # Store
        x_array[:,[k+1]] = x

    # Return
    return x_array



def PlotResidualsOfMethods(A,b,x0,kmax):
    # Set range of k
    k_range = np.arange(kmax+1);

    # Get approximations of methods
    SD_array = SDmethod(A,b,x0,kmax)
    CG_array = CGmethod(A,b,x0,kmax)
    
    # Initialize vectors for
    # 2-norms of residuals of methods
    SD_res = np.zeros([kmax+1,1])
    CG_res = np.zeros([kmax+1,1])
    
    # Compute 2-norms of residuals
    for k in k_range:
        SD=b-A@SD_array[:,[k]]
        CG=b-A@CG_array[:,[k]]
        SD_res[k] = np.linalg.norm(SD)
        CG_res[k] = np.linalg.norm(CG)

    
    
    
    # Preparing figure, using Object -Oriented (OO) style; see:
    # https://matplotlib.org/stable/tutorials/introductory/quick_start.html 
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("||b - A x_k||")
    ax.set_title("Convergence of norms residuals")
    ax.grid(True)
    
    # Plot using following:
    ax.plot(k_range , SD_res , marker="o", label="SD method",linestyle="--")
    ax.plot(k_range, CG_res, marker='o', label="CG method")



    # Add legend
    ax.legend();    
    
    return fig, ax, SD_res, CG_res





def CGNEmethod(A,b,x0,kmax):
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    z_array = np.zeros([np.shape(x0)[0],kmax+1])
    r = np.zeros([np.shape(x0)[0],kmax+1])
    p = np.zeros([np.shape(x0)[0],kmax+1])
    
    #Initial r, p and z
    r[:,[0]] = b
    
    p[:,[0]] = A.T@r[:,[0]]
    
    z_array[:,[0]] = A.T@r[:,[0]]
    
    
    # Set range of k
    k_range = np.arange(kmax);
    
    #CGNE loop
    for k in k_range:
        alpha = (z_array[:,k]@z_array[:,[k]])/(((A@p[:,[k]]).T)@(A@p[:,[k]]))
        x_array[:,[k+1]] = x_array[:,[k]] + alpha*p[:,[k]]
        r[:,[k+1]] = r[:,[k]] - alpha*A@p[:,[k]]
        z_array[:,[k+1]] = A.T@r[:,[k+1]]
        beta = (z_array[:,k+1]@z_array[:,[k+1]])/(z_array[:,k]@z_array[:,[k]])
        p[:,[k+1]] = z_array[:,[k+1]] + beta*p[:,[k]]
        

    # Return
    return x_array, z_array





def PowerIterationMethod(A,v0,kmax):
    
    # Initialize
    m = np.shape(v0)[0]
    v_array = np.zeros([m, kmax])
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    
    
    k_range = np.arange(0,kmax)
    
    for k in k_range:
        if k ==0:
            w = A @ v0
        else:
            w = A @ v_array[:,[k-1]]
        v_array[:,[k]] = w / np.linalg.norm(w)
        eigval_array[k] = v_array[:,k] @ A @ v_array[:,[k]]
        r_array[k] = np.linalg.norm(eigval_array[k]*v_array[:,[k]] - A@v_array[:,[k]])
        
    
    return v_array, eigval_array, r_array






def RayleighQuotientIteration(A,v0,kmax):
    
    # Initialize
    v_array = np.zeros([np.shape(v0)[0],kmax])
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    
    
    k_range = np.arange(0,kmax)
    
    for k in k_range:
        if k == 0:
            eigval_array[k] = v0.T @ A @ v0
            a = A-eigval_array[0]*np.eye(np.shape(A)[0])
            b = v0
            w = np.linalg.solve(a, b)

        else:
            a = A-eigval_array[k-1]*np.eye(np.shape(A)[0])
            b = v_array[:,[k-1]]
            w = np.linalg.solve(a, b)
            
        v_array[:,[k]] = w / np.linalg.norm(w)
        eigval_array[k] = v_array[:,k] @ A @ v_array[:,[k]]
        
        r_array[k] = np.linalg.norm(eigval_array[k]*v_array[:,[k]] - A@v_array[:,[k]])
    


    return v_array, eigval_array, r_array

def CGAATmethod(A,b,y0,kmax):

    #Initialize
    n = np.shape(y0)[0]
    x_array = np.zeros([n, kmax+1])
    y_array = np.zeros([n, kmax+1])
    
    y_array = CGmethod(A@A.T, b, y0, kmax)
    x_array = A.T @ y_array
    
    # Return
    return x_array, y_array



def EffPowerIterMethod(A,v0,kmax,p):
    # Initialize
    n = np.shape(v0)[0]
    v_array = np.zeros([n,kmax])
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    z_array = np.zeros([n,kmax])

    v_array[:,[0]] = v0
    
    k_range = np.arange(0, kmax)
    
    for k in k_range:
        if k == 0:
            v = v0
        else:
            v = v_array[:,[k-1]]
        z_array[:,[k]] = A @ v
        e = v - (z_array[:,[k]] / (v.T@z_array[:,[k]]))
        
        y_array = np.zeros([n,1])
        
        for j in np.arange(0,n):

            largest_p_in_e = np.zeros(p)
            order = np.argsort(np.abs(e.T)[0])
            for i in range(p):
                largest_p_in_e[i] = e[order[-(i+1)]]
                
            if e[j] in largest_p_in_e:
                y_array[j] = z_array[j,k]/(v.T@z_array[:,[k]])

            else:
                y_array[j] = v[j]

        v_array[:, [k]] = y_array/np.linalg.norm(y_array)
        eigval_array[k] = v_array[:,k]@A@v_array[:,[k]]
        r_array[k] = np.linalg.norm(eigval_array[k]*v_array[:,[k]] - A@v_array[:,[k]])
        


    return v_array, eigval_array, r_array, z_array 





























