import numpy as np

class gd_1d:
    
    def __init__(self, fn_loss, fn_grad):
        self.fn_loss = fn_loss
        self.fn_grad = fn_grad
        
    def pv(self, x_1_init, x_2_init, n_iter, eta, tol):
        x = [x_1_init,x_2_init]
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))

        for i in range(n_iter):
            if g_mag < tol or np.isnan(g_mag):
                break
            g = self.fn_grad(x[0],x[1])
            g_mag = np.sum(np.square(g))
            x[0] += -eta * g[0]
            x[1] += -eta * g[1]
            x=[x[0],x[1]]
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)
            
        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x_1 = {} and x_2 = {}'.format(i, loss_this, x[0],x[1]))
            no_steps = i
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        self.no_steps = no_steps
 
        
    def momentum(self, x_1_init, x_2_init, n_iter, eta, tol, alpha):
        x = [x_1_init,x_2_init]

        loss_path = []
        x_path = []

        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))
        nu = [0,0]

        for i in range(n_iter):
            g = self.fn_grad(x[0],x[1])
            g_mag = np.sum(np.square(g))
            if g_mag < tol or np.isnan(g_mag):
                break

            nu[0] = alpha * nu[0] + eta * g[0]
            nu[1] = alpha * nu[1] + eta * g[1]
            
            x[0] += -nu[0]
            x[1] += -nu[1]
            
            x=[x[0],x[1]]
            
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)

        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x_1 = {} and x_2 = {}'.format(i, loss_this, x[0],x[1]))
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        
    def nag(self, x_1_init,x_2_init, n_iter, eta, tol, alpha):
        x = [x_1_init,x_2_init]
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))
        nu = [0,0]

        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g = self.fn_grad(x[0] - nu[0]*mu,x[1] - nu[1]*mu)
            g_mag = np.sum(np.square(g))
            if g_mag < tol or np.isnan(g_mag):
                break

            nu[0] = alpha * nu[0] + eta * g[0]
            nu[1] = alpha * nu[1] + eta * g[1]
            
            x[0] += -nu[0]
            x[1] += -nu[1]
            
            x=[x[0],x[1]]
            
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)

        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x = {} and x_2 = {}'.format(i, loss_this, x[0],x[1]))
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        

        
    def pv_tabulation(self, x_1_init, x_2_init, n_iter, eta, tol):
        x = [x_1_init,x_2_init]

        loss_path = []
        x_path = []

        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))

        for i in range(n_iter):
            if g_mag < tol or np.isnan(g_mag):
                break
            g = self.fn_grad(x[0],x[1])
            g_mag = np.sum(np.square(g))
            x[0] += -eta * g[0]
            x[1] += -eta * g[1]
            x=[x[0],x[1]]
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)

        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            no_steps = i

        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        self.no_steps = no_steps
    
    def momentum_tabulation(self, x_1_init, x_2_init, n_iter, eta, tol, alpha):
        x = [x_1_init,x_2_init]

        loss_path = []
        x_path = []

        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))
        nu = [0,0]

        for i in range(n_iter):
            g = self.fn_grad(x[0],x[1])
            g_mag = np.sum(np.square(g))
            if g_mag < tol or np.isnan(g_mag):
                break

            nu[0] = alpha * nu[0] + eta * g[0]
            nu[1] = alpha * nu[1] + eta * g[1]
            
            x[0] += -nu[0]
            x[1] += -nu[1]
            
            x=[x[0],x[1]]
            
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)

        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            no_steps = i
        
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        self.no_steps = no_steps
        
    def nag_tabulation(self, x_1_init,x_2_init, n_iter, eta, tol, alpha):
        x = [x_1_init,x_2_init]
        
        loss_path = []
        x_path = []
        
        x_path.append(x)
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag = np.sum(np.square(g))
        nu = [0,0]

        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g = self.fn_grad(x[0] - nu[0]*mu,x[1] - nu[1]*mu)
            g_mag = np.sum(np.square(g))
            if g_mag < tol or np.isnan(g_mag):
                break

            nu[0] = alpha * nu[0] + eta * g[0]
            nu[1] = alpha * nu[1] + eta * g[1]
            
            x[0] += -nu[0]
            x[1] += -nu[1]
            
            x=[x[0],x[1]]
            
            x_path.append(x)
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)

        if np.isnan(g_mag):
            print('Exploded')
        elif np.abs(g_mag) > tol:
            print('Did not converge')
        else:
            no_steps = i
        
        self.loss_path = np.array(loss_path)
        self.x_path = np.array(x_path)
        self.no_steps = no_steps