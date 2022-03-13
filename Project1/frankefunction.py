import numpy  as np

class Franke:
    def __init__(self, mean: float = 0.0, deviation: float = 0.0, seed: int = None) -> None:
        """
        The Franke function class
        """
        
        self.mu     = mean
        self.sigma  = deviation
        self.seed   = seed
        self.noise  = False
        self.steplength = None; self.steps = None;
        
        if seed != None:
            np.random.seed(seed)
            # rng = np.random.default_rng(2021)
            # rng = np.random.RandomState(2021)
        
    def initialize_array(self, random = False, precision = 0.05):
        steplength = precision
        steps = int(1/steplength)
        self.steplength = steplength; self.steps = steps
        if random == False:
            print("random: ", random, "steps: ", steps, "steplength: ", steplength)
            x_array = np.arange(0, 1, steplength)
            y_array = np.arange(0, 1, steplength)
        else:
            print("random: ", random, "steps: ", steps, "steplength: ", steplength)
            x_array = np.sort(np.random.rand(steps))
            y_array = np.sort(np.random.rand(steps))
        return x_array, y_array
        
    def franke_function(self, x, y, noise = False):
        n= len(x)
        self.noise = noise
        
        term1 =  0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 =  0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 =  0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        
        if noise:
            # normal_dist = np.random.normal(self.mu, self.sigma, n)
            normal_dist = np.random.normal(self.mu, self.sigma, (n,n))
            return term1 + term2 + term3 + term4 + normal_dist, np.var(normal_dist)
        else:
            return term1 + term2 + term3 + term4
        
    def plot_contour(self, x, y, z, title= False):
        # from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt
        
        # plt.close()
        fig  = plt.figure()
        ax   = fig.gca(projection= '3d')
        
        # ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
        cset = ax.contour(x, y, z, zdir='x', cmap=cm.coolwarm)
        cset = ax.contour(x, y, z, zdir='y', cmap=cm.coolwarm)
        cset = ax.contour(x, y, z, zdir='z', cmap=cm.coolwarm)
        
        if title == False:
            if self.noise:
                plt.title(f"Franke function (noisey $\sigma$: {self.sigma})")
            else:
                plt.title("Franke function (no noise)")
        else:
            plt.title(title)
        
        # Customize the z axis
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.draw()
        # fig.savefig(f"plot_franke_{hex(id(Franke))}", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_3D(self, x, y, z, title= False):
        # from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt
        
        # plt.close()
        fig  = plt.figure()
        ax   = fig.gca(projection= '3d')
        
        surf = ax.plot_surface(x, y, z, cmap = cm.coolwarm, \
                                linewidth = 0, antialiased = False)
        if title == False:
            if self.noise:
                plt.title(f"Franke function (noisey $\sigma$: {self.sigma})")
            else:
                plt.title("Franke function (no noise)")
        else:
            plt.title(title)
        
        # Customize the z axis
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps vlaues to colors.
        cbaxes = fig.add_axes([0.15, 0.2, 0.05, 0.5])
        fig.colorbar(surf, shrink= 0.5, aspect= 5, cax= cbaxes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.draw()
        # fig.savefig(f"plot_franke_{hex(id(Franke))}", dpi=300, bbox_inches='tight')
        
        plt.show()