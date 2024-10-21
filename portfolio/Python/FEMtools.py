"""
MATH3036 CW1 FEMtools module

@author: Thea Hellen
"""

import numpy as np



def AssembleFEMmatrix1(N):
    """
    Returns a numpy array of the 
    Petrov-Galerkin FEM matrix for the bilinear form
       b(u,v) := \int_0^1 u' v dx 
    using FEM trial space 
       Uh := continuous piecewise linears, and being = 0 for x=0, 
    with basis { phi_j } being the hat functions (node-wise), 
    and using the FEM test space
       Vh := discontinuous piecewise constants
    with basis { psi_i } being indicator function (element-wise),
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
        
    Returns
    -------
    A : numpy.ndarray, shape (N,N)
        Array containing the Petrov-Galerkin FEM matrix.
    """
    
    N = int(N)

    # Assemble matrix
    A = np.diag(np.ones(N))         \
      - np.diag(np.ones(N-1),-1)
        
    return A



def AssembleFEMvector1(N,f):
    """
    Returns a numpy array of the 
    FEM vector for the linear form
       l(v) := \int_0^1 f v dx 
    using FEM test space
       Vh := discontinuous piecewise constants
    with basis { psi_i } being indicator function (element-wise),
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
    f : function
        Input function.
        
    Returns
    -------
    b : numpy.ndarray, shape (N,)
        Array containing the FEM vector.
    """
    
    N = int(N)
    
    # Mesh mid points (center of each element)
    h = 1/N;
    xc = h*np.arange(N) + h/2


    # Assemble vector (using mid-point rule)
    b = h*f(xc)
    
    return b


def AssembleFEMmatrix2(N):
    N = int(N)
    # Assemble matrix (coincidentally(!) the same as AssembleFEMmatrix1)
    A =    np.diag(np.ones(N))         \
         - np.diag(np.ones(N-1),-1)
         
    return A


def AssembleFEMvector2(N,f):
    N = int(N)
    # Mesh mid points (center of each element)
    h = 1/N;
    xc_l = h*np.arange(0,N-1) + h/2   # All midpoints, except right-most
    xc   = h*np.arange(0,N)   + h/2   # All midpoints

    # Assemble vector (using mid-point rule)
    b = h*1/2*( f(xc) + np.r_[0,f(xc_l)] )
    
    return b


def AssembleFEMmatrix3(N):
    """
    Returns a numpy array of the 
    Galerkin FEM matrix based on the weak form with bilinear form
        b(u,v) := \int_0^1 u' v' dx 
    and linear form
        l(v) := \int_0^1 f v dx + g_1 v(1)
    using FEM trial space for second-order differential equation.
       V_h := continuous piecewise linears, equal to 0 at x=0, 
    with basis { phi_i } being the hat functions (node-wise), 
    and using the FEM test space
       U_h = V_h due to Galerkin discreditisation where
       u_h in U_h := \sum_j=1^N u_j phi_j(x)
    with basis { phi_i } as above and where u_j are coefficients
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
        
    Returns
    -------
    A : numpy.ndarray, shape (N,N)
        Array containing the Galerkin FEM matrix.

    """
    N = int(N)
    A = np.diag(2*np.ones(N))   \
        -np.diag(np.ones(N-1),-1)   \
            -np.diag(np.ones(N-1),+1)
    
    A[-1,-1]=1
    
    h=1/N
    
    A=(1/h)*A
    
    return A



def AssembleFEMvector3(N,f,g1):
    """
    Returns a numpy array of the 
    FEM vector based on the weak form with bilinear form
        b(u,v) := \int_0^1 u' v' dx 
    and linear form
        l(v) := \int_0^1 f v dx + g_1 v(1)
    using FEM trial space for second-order differential equation, 
    approximated using the composite mid-point rule.
       V_h := continuous piecewise linears, equal to 0 at x=0, 
    with basis { phi_i } being the hat functions (node-wise), 
    and using the FEM test space
       U_h = V_h due to Galerkin discreditisation where
       u_h in U_h  := \sum_j=1^N u_j phi_j(x)
    with basis { phi_i } as above and where u_j are coefficients
    for a uniform mesh with N+1 grid-nodes.  
    
    Parameters
    ----------
    N : integer
        N+1 is the number of nodes in the mesh.
    f : function
        Input function.
    g1 : real number
        g1 is the initial condition at u'(1)
        
    Returns
    -------
    b : numpy.ndarray, shape (N,)
        Array containing the FEM vector.

    """
    N = int(N)
    
    h=1/N
    
    xc_l = h*np.arange(0,N-1) + h/2
    xc_r = h*np.arange(0,N-1) + 3*h/2
    xc   = h*np.arange(0,N)   + h/2
    
    bi = h*1/2*( f(xc_l) + f(xc_r) )
    bn = h*1/2*f(xc[-1]) + g1
    
    b = np.r_[bi,bn]
    
    return b



def AssembleFEMmatrix4(N,alpha,beta):
    """
    As N increases we obtain a smoother curve, i.e. as N tends towards infinity
    approximations converge towards the actual u. At N=1 we get a straight line,
    which gradually curves towards the solution as N increases. At around N=10 we
    get a reasonable and accurate approximation, we get a reasonable approximation 
    at N=7. 
    
    As alpha tends towards zero we get a line which oscillates from roughly 0-2, with a
    gradual slope upwards. At alpha=0.01-0.02 we start to see these oscillations
    along the slope. As alpha tends towards 1 we see the slope shift from skewing
    to the right to skewing to the left. Then as alpha increases beyond 1, we see 
    the peak of the slope decrease until the peak is at the starting point, i.e.
    the skew carries on shifting to the left until it is beyond (0,1).
    
    We also see this skew shift when beta is slightly less than alpha, when
    beta=-alpha we get a straight slope down, when beta<-alpha the graph
    tends towards a reciprocal graph. When beta = -2N*alpha, A is singular and so
    there is no solution, when beta < -2N*alpha we get a graph that starts to oscillate
    and when beta << -2N*alpha we get a graph that oscillates between roughly 0 and -2 
    with a downwards slope, which stays the same even as beta tends towards 
    -infinity. As beta increases we get something similar but in the reverse, the
    skew shifts to the right, and beyond beta = 2N*alpha the graph starts to
    oscillate, this time tending towards oscilations between roughly 0 and 2 in an 
    upward slope.
    """
    N = int(N)
    if alpha<0:
        raise ValueError('alpha must be greater than or equal to 0')
    Aa = np.diag(2*np.ones(N))   \
    -np.diag(np.ones(N-1),-1)   \
        -np.diag(np.ones(N-1),+1)
        
    Aa[-1,-1]=1
    
    Ab = np.diag(np.zeros(N))   \
    -np.diag(np.ones(N-1),-1)   \
        +np.diag(np.ones(N-1),+1)
        
    Ab[-1,-1]=1
    
    h=1/N
    
    A=(alpha/h)*Aa + (beta/2)*Ab
    
    return A



def AssembleMatrix_M(N):
    N = int(N)
    M = np.diag(4*np.ones(N-1))   \
        +np.diag(np.ones(N-2),-1)   \
            +np.diag(np.ones(N-2),+1)
            
    h=1/N
    
    M=(h/6)*M
    
    return M



def AssembleVector_u0(N,u0_func):
    N = int(N)
    h=1/N
    
    xc_l = h*np.arange(0,N-1) + h/2
    xc_r = h*np.arange(0,N-1) + 3*h/2



    b = h*1/2*( u0_func(xc_l) + u0_func(xc_r) )

    
    M = AssembleMatrix_M(N)
    
    u0_vec = np.linalg.solve(M, b)
    
    return u0_vec


    
def AssembleMatrix_K(N):
    N = int(N)
    K = np.diag(2*np.ones(N-1))   \
        -np.diag(np.ones(N-2),-1)   \
            -np.diag(np.ones(N-2),+1)
    h=1/N
    
    K=(1/h)*K
    
    return K


def HeatEqFEM(tau,alpha,Ntime,N,u0_func):
    N = int(N)
    Ntime=int(Ntime)
    u_array = np.zeros((Ntime+1, N-1))
    
    M = AssembleMatrix_M(N)
    
    K = AssembleMatrix_K(N)
    
    u_array[0,:] = AssembleVector_u0(N, u0_func)
    
    for i in range(Ntime):
        u_array[i+1,:] = np.linalg.solve(M, (M-(tau*alpha*K))@u_array[i,:])
    
    return u_array



def GramSchmidtQR(A):
    if type(A)!=np.ndarray:
        raise TypeError("A must be a numpy array")
    # Initialize
    Q = np.zeros(np.shape(A))
    
    # Number of columns
    n = np.shape(A)[1]
    
    # Initialize
    R = np.zeros([n,n])

    # Gram-Schmidt loop
    for j in np.arange(n):
        v = A[:,j]
        for i in np.arange(j):
            R[i,j] = Q[:,i].T @ A[:,j]
            v = v - R[i,j] * Q[:,i]
        R[j,j] = np.linalg.norm(v)
        Q[:,j] = v / R[j,j]
        
    # Return Q,R
    return Q,R



def ModGramSchmidtQR(A):
    if type(A)!=np.ndarray:
        raise TypeError("A must be a numpy array")
    # Initialize
    Q = np.zeros(np.shape(A))
    V = np.copy(A)
    
    # Number of columns
    n = np.shape(A)[1]
    
    # Initialize
    R = np.zeros([n,n])
    
    # Gram-Schmidt loop
    for i in np.arange(n):
        
        for j in np.arange(i+1):
            R[j,i] = Q[:,j].T @ V[:,i]
            V[:,i] = V[:,i] - R[j,i] * Q[:,j]
        
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i] / R[i,i]



    # Return Q,R
    return Q,R



def ModReorthGramSchmidtQR(A):
    if type(A)!=np.ndarray:
        raise TypeError("A must be a numpy array")
    # Initialize
    Q = np.zeros(np.shape(A))
    V = np.copy(A)
    
    # Number of columns
    n = np.shape(A)[1]
    
    # Initialize
    R = np.zeros([n,n])
    
    for j in np.arange(n):
        for i in np.arange(j):
            R[i,j] = V[:,i].T @ V[:,j]
            V[:,j] = V[:,j] - R[i,j]*V[:,i]
        for i in np.arange(j):
            s=V[:,i].T@V[:,j]
            V[:,j]=V[:,j]-s*V[:,i]
            R[i,j]=R[i,j]+s
        R[j,j]=np.linalg.norm(V[:,j])
        V[:,j]=V[:,j]/R[j,j]
        Q[:,j]=V[:,j]


    # Return Q,R
    return Q,R
