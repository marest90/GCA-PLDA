import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import sys

sys.path.insert(0, '/home/mariel/dcaplda-repo/DCA-PLDA')
import dca_plda.calibration as calibration


def normal_1d(mu, pres, x, n):
    """ Unidimensional normal with mu mean value and sigma the dispersion """
    """ All three elements must be floats """

    #if n == 0:
    #    print('uni-dif=', x - mu)
    #    print('pres=', pres)
    #    print('uni-exp=', -pres / 2 * (x - mu) **2)

    
    lognormal = np.log(np.sqrt(abs(pres)) / np.sqrt(2 * np.pi)) - pres/2 * (x - mu) **2


    return lognormal



def normal_kd(theta_list, pres_list, x_list, n):
    """ Multivariate normal of dimension k, the lenght of x  """
    """ The theta is the vector of mean values while sigma is the precision matrix """
    """ All three elements must be lists, the sigma is a list containing the elements in the diagonal, only diagonal allowed """

    x = np.asarray(x_list)
    theta = np.asarray(theta_list)
    diag_array = np.asarray(pres_list)
    pres = np.diag(diag_array)
    #print(x)
    k = len(x)
    dif = np.subtract(x, theta)
    #print('dif=', dif)
    #pres = np.linalg.inv(sigma)
    #if np.linalg.det(pres) < 0:
    #    print('DET UNDER ZERO!!')
    #    print('exponencial=', -1/2 * (np.matmul(dif, np.matmul(pres, dif))))

        #print('pres=', pres)

    #if n==0:
    #    print('exponencial=', -1/2 * (np.matmul(dif, np.matmul(pres, dif))))
    #    print('dif=', dif)
    #    print('pres=', pres_list)

    
    lognormal = np.log(np.sqrt(abs(np.linalg.det(pres))) / np.sqrt( (2 * np.pi) **k)) -1/2 * (np.matmul(dif, np.matmul(pres, dif))) 


    return lognormal


def ink(k, n, x, s, l, params):

    xn = x[n]
    sn = s[n]
    ln = l[n]
    wk = params['w'][-1][k]
    thetak = params['theta'][-1][k]
    phik = params['phi_pres'][-1][k]
    muTk = params['muT'][-1][k]
    muNk = params['muN'][-1][k]
    lambk = params['lamb'][-1][k]

    #print(phik)
    #print(thetak)
    #print(xn)

    N1 = normal_kd(thetak, phik, xn, n)
    N2 = normal_1d(muTk, float(lambk), sn, n)
    N3 = normal_1d(muNk, float(lambk), sn, n)

    logi = np.log(wk) + N1 + ln *N2 + (1-ln) * N3

    return wk, N1, N2, N3, ln, logi


def zeta(i_nk):

    n = len(i_nk)
    k = len(i_nk[0])
    z_nk = np.zeros((n,k))


    for nn in range(n):
        i_sum = np.log(sum(np.exp(i_nk[nn])))
    
        z_nk[nn] = np.exp(i_nk[nn] - i_sum)

    return z_nk

def accumulated_params(z, l, s, x):

    n = len(z)
    k = len(z[0])
    auxn1 = np.zeros((n,k))
    auxn0 = np.zeros((n,k))
    auxr1 = np.zeros((n,k))
    auxr0 = np.zeros((n,k))
    auxr = np.zeros((n,k))

    for nn in range(n):
        auxn1[nn] = z[nn] * l[nn]
        auxn0[nn] = z[nn] * (1 - l[nn])
        auxr1[nn] = z[nn] * l[nn] * s[nn]
        auxr0[nn] = z[nn] * (1 - l[nn]) * s[nn]
        auxr[nn] = z[nn] * s[nn] * s[nn]

    nx = len(x[0])
    auxrx = np.zeros((k,nx))

    for kk in range(k):
        count = np.zeros(nx)
        for nn in range(n):
            #print(cuenta, z[nn,kk]*x[nn])
            count = count + z[nn,kk]*x[nn]
        auxrx[kk] = count

    auxrxx = np.zeros((k,nx))

    for kk in range(k):
        count2 = np.zeros(nx)
        for nn in range(n):
            #print(cuenta, z[nn,kk]*x[nn])
            count2 = count2 + z[nn,kk]*x[nn]*x[nn]
        auxrxx[kk] = count2


    N1 = np.sum(auxn1,axis=0) 
    N0 = np.sum(auxn0,axis=0)
    r1 = np.sum(auxr1,axis=0) 
    r0 = np.sum(auxr0,axis=0) 
    r = np.sum(auxr,axis=0) 
    rx = auxrx 
    rxx = auxrxx

        

    return N1, N0, r1, r0, r, rx, rxx


def Q_func(params, N1, N0, r1, r0, r, rx, rxx, N, D):


    phi_mat = []
    for kk in range(len(params['phi_pres'][-1])):
        phi_mat.append(np.diag(params['phi_pres'][-1][kk]))

    #print('phi=', phi_mat)
    #print('lamb', params['lamb'][-1])

    Q1 = - N/2 * (1+ 2*D) * np.log(2*np.pi) + np.log(params['rho'][-1]) * sum(N1) 
    Q2 = np.log(1-params['rho'][-1]) * sum(N0) + 0.5* sum((N1 + N0) * (2*np.log(params['w'][-1]) + np.log(params['lamb'][-1]) + np.log(abs(np.linalg.det(phi_mat))))) 
    Q3 = -0.5* sum(params['lamb'][-1] * (r -2* r1 * params['muT'][-1] -2* r0 * params['muN'][-1] + N1 * params['muT'][-1] * params['muT'][-1] + N0 * params['muN'][-1] * params['muN'][-1])) 
    Q4 = -0.5 * sum(sum(params['phi_pres'][-1] *(rxx - 2 * rx * params['theta'][-1] + np.transpose((N1+N0) * np.transpose(params['theta'][-1] * params['theta'][-1]))))) 
    Q = Q1 + Q2 + Q3 + Q4

    return Q/N, Q1/N, Q2/N, Q3/N, Q4/N

def param_update(params, N1, N0, r1, r0, r, rx, rxx, kappa=0):

    k = len(N1)
    n = len(params['theta'][-1][0])
    muT = np.zeros((k))
    muN = np.zeros((k))
    lamb = np.zeros((k))
    theta = np.zeros((k,n))
    phi = np.zeros((k,n))



    rho = sum(N1) / sum(N1+N0)
    w = (N1+N0) / sum(N1+N0)
    
    for kk in range(k):
        muT[kk] = r1[kk] / N1[kk]
        muN[kk] = r0[kk] / N0[kk]
        lamb[kk] = (N1[kk] + N0[kk]) / (r[kk] -2* r1[kk] * params['muT'][-1][kk] -2* r0[kk] * params['muN'][-1][kk]+ N1[kk] * params['muT'][-1][kk] * params['muT'][-1][kk] + N0[kk] * params['muN'][-1][kk] * params['muN'][-1][kk])
        theta[kk] = rx[kk] / (N1[kk] + N0[kk] + kappa)
        phi[kk] = np.divide((N1[kk] + N0[kk]) * np.ones(n), rxx[kk] - 2 * rx[kk] * params['theta'][-1][kk] + (N1[kk] + N0[kk])*  (params['theta'][-1][kk] * params['theta'][-1][kk]))


    return rho, w, muT, muN, lamb, theta, phi


def fCMLG2(score_list, lamb, muT, muN):

    f = []
    for s in score_list:
        f.append(lamb * (muT - muN) * (s - (muT + muN)/2))

    return f

def fCMLG(score_list, key_list):

    index_e = []
    index_d = []
    for i in range(len(key_list)):
        if key_list[i] == 1:
            index_e.append(i)
        else:
            index_d.append(i)

    s_e = [score_list[j] for j in index_e]
    s_d = [score_list[j] for j in index_d]

    m_e = sum(s_e) / len(s_e)
    m_d = sum(s_d) / len(s_d)

    ss_e = []
    ss_d = []

    for i in range(len(s_e)):
        ss_e.append((s_e[i]-m_e)*(s_e[i]-m_e))

    for i in range(len(s_d)):
        ss_d.append((s_d[i]-m_d)*(s_d[i]-m_d))

    alpha = len(s_e) / (len(s_e) + len(s_d))
    print('alpha=',alpha)

    v = alpha/len(s_e) * sum(ss_e) + (1-alpha)/len(s_d) * sum(ss_d)

    a = (m_e - m_d)/v
    b = -1* a * (m_e + m_d)/2

    f = []
    for i in range(len(score_list)):
        f.append(a * score_list[i] + b)

    return f


def fGCA(params, s, x, n):

    xn = x[n]
    sn = s[n]
    k = params.param_dict['k']
    w = params.param_dict['w'][-1]
    theta = params.param_dict['theta'][-1]
    phi = params.param_dict['phi_pres'][-1]
    muT = params.param_dict['muT'][-1]
    muN = params.param_dict['muN'][-1]
    lamb = params.param_dict['lamb'][-1]


    #print(k)
    a = np.zeros(k)
    b = np.zeros(k)
    #print(a)

    for i in range(k):
        a[i] = np.exp(np.log(w[i]) + normal_kd(theta[i], phi[i], xn,n) + normal_1d(muT[i], lamb[i], sn,n))
        b[i] = np.exp(np.log(w[i]) + normal_kd(theta[i], phi[i], xn,n) + normal_1d(muN[i], lamb[i], sn,n))

    
        #a[i] = w[i] * np.exp(normal_kd(theta[i], phi[i], xn,n)) * np.exp(normal_1d(muT[i], lamb[i], sn,n))
        #b[i] = w[i] * np.exp(normal_kd(theta[i], phi[i], xn,n)) * np.exp(normal_1d(muN[i], lamb[i], sn,n))



    f = np.log(sum(a)/sum(b))
    #print(f)

    return f


def fGCAj(params, s, x, n, j):

    xn = x[n]
    sn = s[n]
    k = params.param_dict['k']
    w = params.param_dict['w'][j]
    theta = params.param_dict['theta'][j]
    phi = params.param_dict['phi_pres'][j]
    muT = params.param_dict['muT'][j]
    muN = params.param_dict['muN'][j]
    lamb = params.param_dict['lamb'][j]

    print(lamb)

    #print(k)
    a = np.zeros(k)
    b = np.zeros(k)
    #print(a)

    for i in range(k):
        a[i] = np.exp(np.log(w[i]) + normal_kd(theta[i], phi[i], xn,n) + normal_1d(muT[i], lamb[i], sn,n))
        b[i] = np.exp(np.log(w[i]) + normal_kd(theta[i], phi[i], xn,n) + normal_1d(muN[i], lamb[i], sn,n))

    
        #a[i] = w[i] * np.exp(normal_kd(theta[i], phi[i], xn,n)) * np.exp(normal_1d(muT[i], lamb[i], sn,n))
        #b[i] = w[i] * np.exp(normal_kd(theta[i], phi[i], xn,n)) * np.exp(normal_1d(muN[i], lamb[i], sn,n))



    f = np.log(sum(a)/sum(b))
    #print(f)

    return f



def calculate_tar_non(score_list, key_list):
    """ Returns tar and non """

    index_e = []
    index_d = []
    for i in range(len(key_list)):
        if key_list[i] == 1:
            index_e.append(i)
        else:
            index_d.append(i)

    s_e = [score_list[j] for j in index_e]
    s_d = [score_list[j] for j in index_d]  

    return np.array(s_e), np.array(s_d)  



def loglike2(params, s, x, l):

    n = len(x)
    k = params.param_dict['k']
    j = len(params.param_dict['rho'])

    lll = []

    for jj in range(j):
        loglikelihood = None
        print('iteracion=', jj)
        #print('w=', params.param_dict['w'][jj])
        i_array = np.zeros((n,k))
        norma = np.zeros((n,k))
        for nn in range(n):
            for kk in range(k):

                xn = x[nn]
                sn = s[nn]
                ln = l[nn]
                wk = params.param_dict['w'][jj][kk]
                thetak = params.param_dict['theta'][jj][kk]
                phik = params.param_dict['phi_pres'][jj][kk]
                muTk = params.param_dict['muT'][jj][kk]
                muNk = params.param_dict['muN'][jj][kk]
                lambk = params.param_dict['lamb'][jj][kk]

                N1 = normal_kd(thetak, phik, xn, nn)
                N2 = normal_1d(muTk, float(lambk), sn, nn)
                N3 = normal_1d(muNk, float(lambk), sn, nn)

                i_array[nn,kk] = np.exp(np.log(wk) + N1 + ln *N2 + (1-ln) * N3)
                norma[nn,kk] = np.exp(np.log(wk) + N1)

        #logrho = np.zeros(n)
        #for nn in range(n):

        #    ln = l[nn]
        #    rho = params.param_dict['rho'][jj]

        #    logrho[nn] = (rho ** ln) * ((1-rho)**(1-ln)) 

        sumak = np.sum(i_array, axis=1)
        sumanormak = np.sum(norma, axis=1)



        aux = 0
        for nn in range(n):
            aux = np.log(sumak[nn]/sumanormak[nn]) + aux  #np.log(logrho[nn] * sumak[nn]) + aux

        loglikelihood = aux / n

        print(loglikelihood)
        lll.append(loglikelihood)


    return lll


def p_l(ln, ptar):

    if ln == 1:
        pl = ptar
    else:
        pl = 1 - ptar

    return pl


def likelihood(params, s, x, l, jj):

    #print('likelihood')

    n = len(x)
    k = params.param_dict['k']
    j = len(params.param_dict['rho'])

    
    loglikelihood = None
    #print('iteracion=', jj)
    #print('w=', params.param_dict['w'][jj])
    i_array = np.zeros((n,k))
    norma = np.zeros((n,k))
    for nn in range(n):
        for kk in range(k):

            xn = x[nn]
            sn = s[nn]
            ln = l[nn]
            wk = params.param_dict['w'][jj][kk]
            thetak = params.param_dict['theta'][jj][kk]
            phik = params.param_dict['phi_pres'][jj][kk]
            muTk = params.param_dict['muT'][jj][kk]
            muNk = params.param_dict['muN'][jj][kk]
            lambk = params.param_dict['lamb'][jj][kk]

            N1 = normal_kd(thetak, phik, xn, nn)
            N2 = normal_1d(muTk, float(lambk), sn, nn)
            N3 = normal_1d(muNk, float(lambk), sn, nn)

            i_array[nn,kk] = np.exp(np.log(wk) + N1 + ln *N2 + (1-ln) * N3)
            #norma[nn,kk] = np.exp(np.log(wk) + N1)
            

    #logrho = np.zeros(n)
    #for nn in range(n):

    #    ln = l[nn]
    #    rho = params.param_dict['rho'][jj]

    #    logrho[nn] = (rho ** ln) * ((1-rho)**(1-ln)) 

    sumak = np.log(np.sum(i_array, axis=1))
    #sumanormak = np.sum(norma, axis=1)



    likeli = np.zeros(n)
    for nn in range(n):
        likeli[nn] = sumak[nn] #/sumanormak[nn]

    return likeli


def CE2(params, s, x, l, ptar):

    #print('likelihood')

    n = len(x)
    k = params.param_dict['k']
    j = len(params.param_dict['rho'])
    CE = []

    for jj in range(j):
        loglikelihood = None
        #print('iteracion=', jj)
        #print('w=', params.param_dict['w'][jj])
        i_array = np.zeros((n,k))
        i_array2 = np.zeros((n,k))
        norma = np.zeros((n,k))
        for nn in range(n):
            for kk in range(k):

                xn = x[nn]
                sn = s[nn]
                ln = l[nn]
                wk = params.param_dict['w'][jj][kk]
                thetak = params.param_dict['theta'][jj][kk]
                phik = params.param_dict['phi_pres'][jj][kk]
                muTk = params.param_dict['muT'][jj][kk]
                muNk = params.param_dict['muN'][jj][kk]
                lambk = params.param_dict['lamb'][jj][kk]

                N1 = normal_kd(thetak, phik, xn, nn)
                N2 = normal_1d(muTk, float(lambk), sn, nn)
                N3 = normal_1d(muNk, float(lambk), sn, nn)

                i_array[nn,kk] = np.exp(np.log(wk) + N1 + ln *N2 + (1-ln) * N3) * p_l(ln, ptar)
                i_array2[nn,kk] = np.exp(np.log(wk) + N1 + N2) * p_l(1, ptar) + np.exp(np.log(wk) + N1 + N3) * p_l(0, ptar)
                #norma[nn,kk] = np.exp(np.log(wk) + N1)
                

        #logrho = np.zeros(n)
        #for nn in range(n):

        #    ln = l[nn]
        #    rho = params.param_dict['rho'][jj]

        #    logrho[nn] = (rho ** ln) * ((1-rho)**(1-ln)) 

        sumak = np.log(np.sum(i_array, axis=1))
        sumak2 = np.log(np.sum(i_array2, axis=1))
        #sumanormak = np.sum(norma, axis=1)



        likeli = np.zeros(n)
        for nn in range(n):
            likeli[nn] = sumak[nn] - sumak2[nn] 

        ce = -1* sum(likeli)
        CE.append(ce)

    return CE



def likelihood_x_pl(params, s, x, l, ptar, jj):

    #print('likelihood_x_pl')

    n = len(x)
    k = params.param_dict['k']
    j = len(params.param_dict['rho'])

    
    loglikelihood = None
    #print('iteracion=', jj)
    #print('w=', params.param_dict['w'][jj])
    i_array = np.zeros((n,k))
    norma = np.zeros((n,k))
    for nn in range(n):
        for kk in range(k):

            xn = x[nn]
            sn = s[nn]
            ln = l[nn]
            wk = params.param_dict['w'][jj][kk]
            thetak = params.param_dict['theta'][jj][kk]
            phik = params.param_dict['phi_pres'][jj][kk]
            muTk = params.param_dict['muT'][jj][kk]
            muNk = params.param_dict['muN'][jj][kk]
            lambk = params.param_dict['lamb'][jj][kk]

            N1 = normal_kd(thetak, phik, xn, nn)
            N2 = normal_1d(muTk, float(lambk), sn, nn)
            N3 = normal_1d(muNk, float(lambk), sn, nn)

            i_array[nn,kk] = np.exp(np.log(wk) + N1 + N2) * p_l(1, ptar) + np.exp(np.log(wk) + N1 + N3) * p_l(0, ptar)
            #norma[nn,kk] = np.exp(np.log(wk) + N1)
            

    #logrho = np.zeros(n)
    #for nn in range(n):

    #    ln = l[nn]
    #    rho = params.param_dict['rho'][jj]

    #    logrho[nn] = (rho ** ln) * ((1-rho)**(1-ln)) 

    sumak = np.log(np.sum(i_array, axis=1))
    #sumanormak = np.sum(norma, axis=1)



    likeli_x_pl = np.zeros(n)
    for nn in range(n):
        likeli_x_pl[nn] = sumak[nn] #/sumanormak[nn]

    return likeli_x_pl


def CE(params, s, x, l, ptar):

    n = len(x)
    j = len(params.param_dict['rho'])
    print('j=',j)

    ce_arg = np.zeros(n)
    num = np.zeros(n)
    den = np.zeros(n)

    CE = []

    for jj in range(j):
        ############################################3
        print('iteracion =', jj)
        for nn in range(n):

            num[nn] = likelihood(params, s, x, l, jj)[nn] * p_l(l[nn], ptar)

            den[nn] = likelihood_x_pl(params, s, x, l, ptar, jj)[nn]
            #print('den=', den[nn])

            ce_arg[nn] = num[nn] - den[nn]

        CE.append(-1 * sum( ce_arg) /n )
        print(CE)

    return CE




class calculate_params():
    
    #yaml_dir = '/home/mariel/dcaplda-repo/GCA-PLDA/speaker_verification/configs/gca_train/param_card.yaml'

    def __init__(self) -> None:
        self.param_dict = {'nx':None, 'k':None, 'w':[], 'muT':[], 'muN':[], 'phi_pres':[], 'theta':[], 'lamb':[], 'rho':[]}

    def load_params(self, yaml_file):
        with open(yaml_file) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        w = params['w']
        nx = params['Nx']
        k = params['k']
        muT = np.repeat(params['mu_T'],k)
        muN = np.repeat(params['mu_N'],k)
        lamb = np.repeat(params['lambda'],k)
        rho = params['rho']

        #only if theta_Sigma and theta_mu are diagonal and all eigenvalues the same
        #phi_diag_elements_inverse = params['phi'][0]**(-1) #this is because we want the dispersion and phi is the precision, since phi is diagonal its inverse is the inverse of the elements of the diagonal
        #phi_n = [params['phi'][0]] * nx #this is a sigma

        #phi_pres = []
        #for kk in range(k):
        #    phi_pres.append(phi_n)

        phi_n = np.repeat(params['phi'], nx)
        phi_pres = np.repeat([phi_n], k, axis=0)
        


        theta_diag_elements_inverse = params['theta_pres']**(-1)
        theta_Sigma = np.diag(np.repeat(theta_diag_elements_inverse,nx))
        theta_mu = np.repeat(params['theta_mean'],nx)

        #print('mu=', len(theta_mu))
        #print('sigma=' ,len(theta_Sigma))
        theta = np.random.multivariate_normal(theta_mu, theta_Sigma, size=k)

  
        if params['w'] == 'uniform_distribution':
            w = np.random.uniform(0,1,k)


        self.param_dict['nx'] = nx
        self.param_dict['k'] = k
        self.param_dict['w'].append(w)
        self.param_dict['muT'].append(muT)
        self.param_dict['muN'].append(muN)
        self.param_dict['lamb'].append(lamb)
        self.param_dict['phi_pres'].append(phi_pres)
        self.param_dict['rho'].append(rho)
        self.param_dict['theta'].append(theta)

    def iter(self, num_it, con, score, key):
        #lambplus = self.param_dict['lamb'][-1]
        #self.param_dict['lamb'].append(lambplus)
        
        n = len(con)
        k = self.param_dict['k']
        Q_list = []

        for j in range(num_it):
            
            i_array = np.zeros((n,k))
            for nn in range(n):
                for kk in range(k):
                    wk, N1, N2, N3, ln, i_array[nn,kk] = ink(kk, nn, con, score, key, self.param_dict)

                    
            zetank = zeta(i_array)

            #print('zetank=', zetank)

            N1, N0, r1, r0, r, rx, rxx = accumulated_params(zetank, key, score, con)

            #print('N1=', N1)
            #print('N0=', N0)
            #print('r1=', r1)
            #print('r0=', r0)
            #print('r=', r)
            #print('rx=', rx)
            #print('rxx=', rxx)

            D = len(con[0]) /2
            N_emb = len(con)

            #print('D=', D, 'N_emb=', N_emb)

            #print('phi=', self.param_dict['phi_pres'][-1])
            print('lamb=', self.param_dict['lamb'][-1])

            #print(modelparam.param_dict['rho'])
            Q, q1, q2, q3, q4= Q_func(self.param_dict, N1, N0, r1, r0, r, rx, rxx, N_emb, D)

            print('Q=',Q)
            print('Qlist=',Q_list)

            rho2, w2, mut2, mun2, lamb2, theta2, phi2 = param_update(self.param_dict, N1, N0, r1, r0, r, rx, rxx)

            self.param_dict['w'].append(w2)
            self.param_dict['muT'].append(mut2)
            self.param_dict['muN'].append(mun2)
            self.param_dict['lamb'].append(lamb2)
            self.param_dict['phi_pres'].append(phi2)
            self.param_dict['rho'].append(rho2)
            self.param_dict['theta'].append(theta2)

            Q_list.append(Q)
            print('############################################################################')
            print('iteracion nro=', j)
        print('Q=', Q_list)
        #print('w=', self.param_dict['w'])
        print('muT=', self.param_dict['muT'])
        print('muN=', self.param_dict['muN'])
        print('lamb=', self.param_dict['lamb'])
        #print('phi=', self.param_dict['phi_pres'])
        #print('rho=', self.param_dict['rho'])
        #print('theta=', self.param_dict['theta'])

    #print(w, w.shape)
    


def funcion(phi):
    return phi*4

class par():

    def __init__(self) -> None:
        self.phi = []

    def load(self):
        yaml_dir = '/home/mariel/dcaplda-repo/GCA-PLDA/speaker_verification/configs/gca_train/param_card.yaml'

        with open(yaml_dir) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        phi = params['k']
        aux = funcion(phi)
        self.phi.append(aux)

    def iter(self):
        phi_plus1 = self.phi[-1]
        self.phi.append(phi_plus1)


#main
#for n

#    for k

#        calculo i(n,k)
    
#    calculo z(n,k) sumando en todos los k
#    calculo (12)

#calculo (15)
#defino nuevos valores iniciales segun (16)
#vuelve a empezar 