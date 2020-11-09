# ====================================================================================== #
# Module for useful functions on the 2D surface of a sphere.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2, arccos, arcsin, pi, arctan
from numba import float64, njit, jit
from numba.experimental import jitclass
from warnings import warn
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import minimize
from itertools import combinations
from .angle import mod_angle, Quaternion
from .utils import ind_to_sub
PRECISION = 2e-7  # b/c of linearity, this is PRECISION * RADIUS distance



# ==================== #
# Standalone functions #
# ==================== #
def plot_unit_sphere(ax, radius=.98):
    """Plot transparent unit sphere.
    
    Parameters
    ----------
    ax : mpl.Axes
    radius : float, .98

    Returns
    -------
    None
    """

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) * radius
    y = np.outer(np.sin(u), np.sin(v)) * radius
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * radius

    # Plot the surface
    ax.plot_surface(x, y, z, alpha=.2, color='k')

def rand(n=1, degree=True):
    """Randomly sample points from the surface of a sphere.

    Parameters
    ----------
    n : int, 1
    degree : bool, True

    Returns
    -------
    randlon : float
    randlat : float
    """

    if degree:
        randlat = arccos(2*np.random.rand(n)-1)/pi*180-90
        randlon = np.random.uniform(-180,180,size=n)
        return randlon, randlat
    randlat = arccos(2*np.random.rand(n)-1)-pi/2
    randlon = np.random.uniform(-pi,pi,size=n)
    return randlon, randlat

def haversine(x, y, r=1):
    """
    Parameters
    ----------
    x,y : tuple
        (phi, theta) azimuthal angle first
    radius : float, 1

    Returns
    -------
    dist : float
    """
    
    dphi = y[0] - x[0]
    # assuming that convention of theta in [0,pi] holds, this shift of coordinate system
    # will vastly enhance accuracy of calculation
    th1 = x[1] - pi/2
    th2 = y[1] - pi/2

    num = np.sqrt((cos(th2) * sin(dphi))**2 + (cos(th1)*sin(th2) -
                  sin(th1) * cos(th2) * cos(dphi))**2)
    den = sin(th1) * sin(th2) + cos(th1) * cos(th2) * cos(dphi)
    return r * arctan2(num, den)

@njit
def jithaversine(x, y):
    """
    Parameters
    ----------
    x,y : tuple
        (phi, theta)

    Returns
    -------
    dist : float
    """

    dphi = y[0] - x[0]
    # assuming that convention of theta in [0,pi] holds, this shift of coordinate system
    # will vastly enhance accuracy of calculation
    th1 = x[1] - pi/2
    th2 = y[1] - pi/2

    num = np.sqrt((cos(th2) * sin(dphi))**2 + (cos(th1)*sin(th2) -
                  sin(th1) * cos(th2) * cos(dphi))**2)
    den = sin(th1) * sin(th2) + cos(th1) * cos(th2) * cos(dphi)
    return arctan2(num, den)

def latlon2angle(*args):
    """
    Parameters
    ----------
    lonlat as one or lon,lat
    """

    if len(args)==2:
        lat, lon = args
        return lat/180*pi, lon/180*pi
    return args[0]/180*pi

def vincenty(point1, point2, a, f, MAX_ITERATIONS=200, CONVERGENCE_THRESHOLD=1e-12):
    """Vincenty's formula (inverse method) to calculate the distance between two points on
    the surface of a spheroid

    Parameters
    ----------
    point1 : twople
        (xy-angle, polar angle). These should be given in radians.
    point2 : twople
        (xy-angle, polar angle)
    a : float
        Equatorial radius.
    f : float
        eccentricity, semi-minor polar axis b=(1-f)*a
    """

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0
    b=(1-f)*a

    U1 = math.atan((1 - f) * math.tan(point1[0]))
    U2 = math.atan((1 - f) * math.tan(point2[0]))
    L = point2[1] - point1[1]
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    return round(s, 6)

def pixelate_voronoi(X, poissd, offset, lonlat=True):
    """Cluster events on the globe using a Voronoi tessellation.

    Parameters
    ----------
    X : pandas.DataFrame or ndarray
        DataFrame: must have 'longitude' and 'latitude' columns in units of degrees.
        ndarray: must be columns in order of lat and lon.
    poissd : PoissonDiscSphere
        Used to determine neighbors.
    offset : float
        Longitudinal offset.
    lonlat : bool, True
        If True, given coordinates are longitude and latitude (and not angles).

    Returns
    -------
    list of lists of indices
        Indices to the given DataFrame grouped by pixel.
    list of indices
        Pixel to which each event belongs sorted in the order of the events in the clusters.
    list of indices
        Pixel to which each event belongs sorted in the original order of events given.
    """

    if type(X) is pd.DataFrame:
        # When a dataframe is passed in, we must account for the possibility that the
        # indices are not nicely sorted such that we can return indices that are
        # consistent with the dataframe's.
        originalix = X.index
        if lonlat:
            lat, lon=X.loc[:,'latitude'].values, X.loc[:,'longitude'].values%360-offset
            pixIx = poissd.closest_neighbor(np.vstack(lonlat2angle(lon, lat)).T, ignore_zero=False)
        else:
            theta, phi = X.loc[:,'latitude'].values, X.loc[:,'longitude'].values
            pixIx = poissd.closest_neighbor(np.vstack((phi, theta)).T, ignore_zero=False)
        uniqIx = np.unique(pixIx)

        clusteredPixIx = []  # pixel ix by cluster order
        pixIxByOriginalIx = np.zeros_like(originalix)  # pix ix in order of original data
        splitix = []

        for i in uniqIx:
            splitix.append( np.array(originalix[np.where(pixIx==i)[0]]) )
            pixIxByOriginalIx[pixIx==i] = i
            clusteredPixIx.extend( [i]*len(splitix[-1]) )

        return splitix, np.array(clusteredPixIx), pixIxByOriginalIx

    if lonlat:
        lat, lon=X[:,0], X[:,1]-offset
        pixIx = poissd.closest_neighbor(np.vstack(lonlat2angle(lon, lat)).T, ignore_zero=False)
    else:
        theta, phi=X[:,0], X[:,1]
        pixIx = poissd.closest_neighbor(np.vstack((phi, theta)).T, ignore_zero=False)
    uniqIx = np.unique(pixIx)

    clusteredPixIx = []
    pixIxByOriginalIx = np.zeros(len(X), dtype=int)  # pix ix in order of original data
    splitix=[]

    for i in uniqIx:
        splitix.append( np.where(pixIx==i)[0] )
        pixIxByOriginalIx[pixIx==i] = i
        clusteredPixIx.extend( [i]*len(splitix[-1]) )
    
    return splitix, np.array(clusteredPixIx), pixIxByOriginalIx

def __max_dist_pair(xy, return_dist=False):
    """Pair of points maximally distance from each other on sphere.

    Parameters
    ----------
    xy : ndarray
        (phi, theta) limited to ranges [0,2*pi] and [-pi/2,pi/2]
    return_dist : bool, False

    Returns
    -------
    tuple
    float (optional)
    """
    
    from .utils import ortho_plane, max_dist_pair2D

    # convert to cartesian coordinates
    xyz = np.vstack([jitSphereCoordinate(*xy_).vec for xy_ in xy])
    
    uxyz = np.unique(xyz, axis=0)
    # project down to plane to get maximally distant pair of points
    com = xyz.mean(0)
    r1, r2 = ortho_plane(com)
    proj = np.vstack((uxyz.dot(r1), uxyz.dot(r2))).T
    ix = max_dist_pair2D(proj)
    
    if return_dist:
        return ix, haversine(xy[ix[0]], xy[ix[1]])
    return ix

def max_geodist_pair(phitheta, force_slow=False, return_dist=False):
    """Find approximately most distant pair of points on surface of the sphere. First,
    collapse points onto 2D plane that is orthogonal to vector to center of mass of
    points. Then, assume that the convex hull contains the most distant pair of points.
    This is different from max_dist_pair2D() because we use the haversine distance for
    this last step.

    Parameters
    ----------
    xy : ndarray
        (x,y) coordinations
    force_slow : bool, False
        Use slow calculation computing entire matrix of pairwise distances.
    return_dist : bool, False

    Returns
    -------
    tuple
        Indices of two max separated points.
    """
    
    from .utils import convex_hull, ind_to_sub, ortho_plane

    if type(phitheta) is list:
        phitheta = np.vstack(phitheta)
    
    # it is faster to do every pairwise computation when the size of the is small
    if force_slow or len(phitheta)<500:
        return _max_dist_pair(phitheta, return_dist)

    xyz = np.zeros((len(phitheta), 3))
    xyz[:,0] = np.sin(phitheta[:,1]) * np.cos(phitheta[:,0])
    xyz[:,1] = np.sin(phitheta[:,1]) * np.sin(phitheta[:,0])
    xyz[:,2] = np.cos(phitheta[:,1])

    # collapse points down to plane orthogonal to center of mass
    mxyz = xyz.mean(0)
    mxyz /= np.linalg.norm(mxyz)
    v1, v2 = ortho_plane(mxyz)

    xy = np.vstack((xyz.dot(v1), xyz.dot(v2))).T
    
    hull = convex_hull(xy, recursive=True)
    dist = pdist(phitheta[hull], jithaversine)
    mxix = ind_to_sub(hull.size, dist.argmax())
    if return_dist:
        return (hull[mxix[0]], hull[mxix[1]]), dist.max()
    return hull[mxix[0]], hull[mxix[1]]
          
def _max_dist_pair(phitheta, return_dist):
    """Slow way of finding maximally distant pair by checking every pair.
    """
    
    assert len(phitheta)>1
    dmat = pdist(phitheta, jithaversine)
    dmaxix = dmat.argmax()
    majix = ind_to_sub(len(phitheta), dmaxix)
    if return_dist:
        return majix, dmat[dmaxix]
    return majix



# ======= #
# Classes #
# ======= #
class PoissonDiscSphere():
    """Generate a random set of points on the surface of a sphere within some specified
    region using a Poisson disc sampling algorithm. This generates a random tiling that is
    much more uniformly spaced than independently sampling the space.
    
    To speed up nearest neighbor searches, the points are all assigned to a coarser grid
    such that only comparisons to the children of the coarse grid are necessary to find
    nearest neighbors.

    This was adapted from the blog article at
    https://scipython.com/blog/poisson-disc-sampling-in-python/ by Christian Hill and
    accessed in March 2017.

    Data members
    ------------
    kCoarse : int
    coarseGrid : ndarray
    coarseNeighbors : list
        Neighbors for each coarse grid point.
    fastSampleSize : int
    height : tuple
    iprint : bool
    kCoarse : int
    nTries : int
    r : float
    rng : np.randomRandomState
    samples : ndarray
        Azimuthal angle comes first.
    samplesByGrid : list
    unif_theta_bounds : tuple
        For sampling neighbors when generating the random tiling.
    width : tuple
    """
    def __init__(self, r,
                 width_bds=(0, 2*pi),
                 height_bds=(-pi/2, pi/2),
                 fast_sample_size=30,
                 n_tries=30,
                 coarse_grid=None,
                 k_coarse=9,
                 iprint=True,
                 rng=None):
        """
        Parameters
        ----------
        r : float
            Angular distance to use to separate random adjacent points.
        width_bds : tuple, (0, 2*pi)
            Longitude range in which to generate random points as radians.
        height_bds : tuple, (-pi/2, pi/2)
            Latitude range in which to generate random points as radians.
        fast_sample_size : int, 30
            Number of points to sample inspect carefully when using heuristic for
            calculating distances for nearest neighbor search. The heuristic uses the
            simple Euclidean distance on angular coordinates to identify a set of close
            neighbors. This is a reasonable approximation far from the poles.
        n_tries : int, 30
            Number of times to try generating random neighbor in annulus before giving up.
            The larger this number, the more uniform the packing is likely to be, but it
            will be proportionally slower.
        coarse_grid : ndarray, None
            Can supply a predefined set of points on which to perform the coarse-grained
            neighbor search. If not provided, this is automatically generated.
        k_coarse : int, 9
            Number of nearest neighbors on the coarse grid to use for fast neighbor
            searching. For the spherical surface about 6 should be good enough for the
            roughly hexagonal tiling, but I find that irregular tiling means having more
            neighbors is a good idea. If this number is too large, many unnecessary
            comparisons will be made with the children of those coarse grids.
        iprint : bool, True
        rng : np.random.RandomState, None
        """

        assert r>0, r
        assert 0<=width_bds[0]<2*pi and 0<=width_bds[1]<2*pi
        assert -pi/2<=height_bds[0]<height_bds[1]<=pi/2

        self.width, self.height = width_bds, height_bds
        self.r = r
        self.nTries = n_tries
        # this determines how far away new neighboring points are allowed to be from from
        # the starting point
        self.unif_theta_bounds = (1+np.cos(r))/2, (1+np.cos(2*r))/2
        self.fastSampleSize = fast_sample_size
        self.iprint = iprint
        self.rng = rng or np.random.RandomState()
        
        # set up grid
        self.kCoarse = k_coarse
        self.samples = np.zeros((0,2))
        self.set_coarse_grid(coarse_grid)
    
    def set_coarse_grid(self, coarse_grid):
        """Assign each sample point to a new coarse grid.
        
        Parameters
        ----------
        coarse_grid : ndarray
        """
        
        if self.iprint:
            print("Setting up coarse grid.")

        self.coarseGrid = coarse_grid
        if not self.coarseGrid is None:
            assert type(coarse_grid) is np.ndarray
            assert coarse_grid.shape[1]==2
            self.preprocess_coarse_grid()
            self.samplesByGrid = [[] for i in self.coarseGrid]

            for i,pt in enumerate(self.samples):
                self.samplesByGrid[self.assign_grid_point(pt)].append(i)
            if self.iprint: print("Coarse grid done.")
        elif self.iprint:
            print("No coarse grid set up.")

    def preprocess_coarse_grid(self):
        """Find the k_coarse nearest coarse neighbors for each point in the coarse grid. Also
        include self in the list which explains the +1.
        """

        coarseNeighbors = []
        for pt in self.coarseGrid:
            coarseNeighbors.append( np.argsort(self.dist(pt,
                                                         self.coarseGrid))[:self.kCoarse+1].tolist() )
        self.coarseNeighbors = coarseNeighbors

    def get_neighbours(self, *args, **kwargs):
        """Deprecated wrapper for neighbors()."""
        
        warn("PoissonDiscSphere.get_neighbours() is now deprecated. Use neighbors() instead.")
        return self.neighbors(*args, **kwargs)

    def neighbors(self, xy,
                  fast=False,
                  top_n=None,
                  apply_dist_threshold=False,
                  return_dist=False):
        """Return top_n neighbors in the grid with option to use fast Euclidean distance
        calculation. All neighbors are guaranteed to be within 2*r*apply_dist_threshold
        though neighbors may not be the closest ones (or sorted) if the fast heuristic is
        used.

        Parameters
        ----------
        xy : ndarray 
            Coordinates for which to find neighbors.
        fast : bool, False
            If True, fast heuristic is used (only when there is no coarse grid!).
        top_n : int, None
            Number of grid points to keep using search with fast distance. 
        apply_dist_threshold : float, False
            If True, that value will be multiplied to the distance window 2*self.r and
            only points within that distance will be returned as potential neighbors.
        return_dist : bool, False
            If True, return solid angle to top_n neighbors that were compared.
            If no coarse grid available, distance to each point in self.samples can be
            returned.

        Returns
        -------
        list of lists
            neighbor_ix
        ndarray
            Solid angle distance to neighbors.
        """

        top_n = top_n or self.fastSampleSize
        threshold = 2 * self.r * apply_dist_threshold
        
        # case where coarse grid is defined
        if not self.coarseGrid is None:
            if len(self.samples):
                # find the closest coarse grid point
                allSurroundingGridIx = self.coarseNeighbors[find_first_in_r(xy, self.coarseGrid, self.r)]
                
                # concatenate list of all neighbors that are within the surrounding grid pts
                neighbors = []
                for ix in allSurroundingGridIx:
                    neighbors += self.samplesByGrid[ix]

                if apply_dist_threshold:
                    d = self.dist(self.samples[neighbors], xy)
                    # select elements satisfying distance criterion
                    neighbors = [neighbors[i] for i, d_ in enumerate(d) if d_<=threshold]
                    d = [d_ for d_ in d if d_<=threshold]

                if return_dist and apply_dist_threshold:
                    return neighbors, np.array(d)  # dist has already been calculated for this
                elif return_dist:
                    d = self.dist(self.samples[neighbors], xy)
                    return neighbors, d
                return neighbors
            if return_dist:
                return [], []
            return []

        if len(self.samples):
            if self.iprint: "No coarse grid available, all pairwise comparisons to find neighbors."

            # find the closest point
            if fast:
                d = self.fast_dist(xy, self.samples)
                neighbors = np.argsort(d)[:top_n]
                if apply_dist_threshold:
                    # calculate true distance for applying cutoff
                    ix = self.dist(self.samples[neighbors], xy) <= threshold
                    neighbors = neighbors[ix]
            else:
                d = self.dist(xy, np.vstack(self.samples))
                neighbors = np.argsort(d)[:top_n]
                if apply_dist_threshold:
                    # can reuse results of distance calculation in this case
                    neighbors = neighbors[d[neighbors]<=threshold]
            if return_dist:
                return neighbors.tolist(), d[neighbors]
            return neighbors.tolist()
        if return_dist:
            return [], []
        return []

    def get_closest_neighbor(self, *args, **kwargs):
        """Deprecated. Use closest_neighbor() instead."""

        warn("Deprecated. Use closest_neighbor() instead.")
        return self.closest_neighbor(*args, **kwargs)

    def closest_neighbor(self, pt, ignore_zero=1e-9):
        """Get closest grid point index for all points given.

        Parameters
        ----------
        pt : tuple
        ignore_zero : float, 1e-9
            Distances smaller than this are ignored for returning the min distance.

        Returns
        -------
        list of ints
            Indices of closest points.
        """

        if pt.ndim==1:
            pt = pt[None,:]
        
        return [self._closest_neighbor(row, ignore_zero) for row in pt]

    def _closest_neighbor(self, pt, ignore_zero=1e-9):
        """Get closest grid point index for a single point.

        Parameters
        ----------
        pt : tuple
        ignore_zero : float, 1e-9
            Distances smaller than this are ignored for returning the min distance. This
            is to account for comparisons that might fail because of floating point
            precision.

        Returns
        -------
        int 
            Index.
        """

        neighborix, distance = self.neighbors(pt, return_dist=True)
        neighborix = np.array(neighborix, dtype=int)

        if ignore_zero and len(neighborix)>0:
            keepix = distance > ignore_zero
            distance = distance[keepix]
            neighborix = neighborix[keepix]
        elif len(neighborix)==0:
            return []
        return neighborix[np.argmin(distance)]

    def closest_neighbor_dist(self, pt, ignore_zero=1e-9):
        """Given a point on the globe, find the sample of closest distance to it. Assumes
        that we have nonzero number of samples.

        Parameters
        ----------
        pt : tuple
            Angular specification of point.
        ignore_zero : float,1e-9
            Distances smaller than this are ignored for returning the min distance.

        Returns
        -------
        float
            Min distance on a unit sphere (units of radians).
        """

        distance = self.dist(pt, self.samples[self.neighbors(pt)])
        if ignore_zero:
            return distance[distance>ignore_zero].min()
        return distance.min()

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?
        It must be no closer than r from any other point: check the cells in its immediate
        neighborhood.
        """
        
        if len(self.samples)>0:
            neighborIx = self.neighbors(pt)

            # handle lists of samples and array of samples differently
            if type(self.samples) is list:
                for ix in neighborIx:
                    if (self.dist(pt, self.samples[ix]) < self.r):
                        return False
            else:
                if (self.dist(pt, self.samples[neighborIx]) < self.r).any():
                    return False
        return True
    
    def get_point(self, refpt, max_iter=10):
        """Try to find a candidate point near refpt to emit in the sample.  We draw up to
        k points from the annulus of inner radius r, outer radius 2r around the reference
        point, refpt. If none of them are suitable (because they're too close to existing
        points in the sample), return False. Otherwise, return the pt in a list.
        """

        sphereRefpt = jitSphereCoordinate(refpt[0]%(2*pi), refpt[1]+pi/2)
        i = 0
        while i < self.nTries:
            # generate a random perturbation of this point within allowed bounds
            pt = sphereRefpt.random_shift_controlled(self.unif_theta_bounds, *self.rng.rand(2))
            # put theta back into same range as this class (since SphereCoordinate uses [0,pi]
            pt = np.array([pt[0], pt[1]-pi/2])
            if not ((self.width[0] < pt[0] < self.width[1] or
                     self.width[0] < (pt[0]-2*pi) < self.width[1]) and
                    self.height[0] < pt[1] < self.height[1]):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return [pt]
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def assign_grid_point(self, pt, fast=False):
        """Find closest coarseGrid point that is near given point for assignment.

        Parameters
        ----------
        pt : ndarray
            Single point.
        fast : bool, False

        Returns
        -------
        int
        """
        
        if fast:
            return np.argmin( self.fast_dist(pt, self.coarseGrid) )
        return np.argmin( self.dist(pt, self.coarseGrid) )

    def sample(self):
        """Draw random samples on the domain width x height such that no two samples are
        closer than r apart. The parameter k determines the maximum number of candidate
        points to be chosen around each reference point before removing it from the
        "active" list.

        Returns
        -------
        ndarray
            sample points
        """

        if not self.coarseGrid is None:
            self.samplesByGrid = [[] for i in self.coarseGrid]
            
        # must account for periodic boundary conditions when generating new points and
        # checking validity of proposed points. By temporarily changing the boundary
        # conditions, we can ensure that such tasks will
        # be performed correctly.
        if self.width[0]>self.width[1]:
            self.width = self.width[0]-2*pi, self.width[1]

        # Pick a random point to start with.
        pt = np.array([self.rng.uniform(*self.width),
                       self.rng.uniform(*self.height)])
        self.samples = [pt]
        if not self.coarseGrid is None:
            self.samplesByGrid[self.assign_grid_point(pt)].append(0)

        # and it is active, in the sense that we're going to look for more points in its
        # neighborhood.
        active = [0]

        # As long as there are points in the active list, keep looking for samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = self.rng.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                self.samples.append(pt[0])
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                if not self.coarseGrid is None:
                    self.samplesByGrid[self.assign_grid_point(pt[0])].append(len(self.samples)-1)
            else:
                # We had to give up looking for valid points near refpt, so remove it from
                # the list of "active" points.
                active.remove(idx)
        
        self.samples = np.vstack(self.samples)
        # When doing a fast distance computation, we cannot take a sample smaller than the
        # size of the system so cap the number of comparisons to the total sample size.
        if len(self.samples)<self.fastSampleSize:
            self.fastSampleSize = len(self.samples)
        if not self.coarseGrid is None:
            assert sum([len(s) for s in self.samplesByGrid])==len(self.samples)
        
        if self.width[0]<0:
            self.samples[:,0] %= 2*pi
            self.width = self.width[0]+2*pi, self.width[1]
        
        return self.samples
    
    @classmethod
    def dist(cls, x, y):
        """Great circle distance. Vector optimized."""

        if x.ndim==2 and y.ndim==1:
            return 2*arcsin( np.sqrt(sin((x[:,1]-y[1])/2)**2 +
                             cos(x[:,1])*cos(y[1])*sin((x[:,0]-y[0])/2)**2) )
        elif x.ndim==1 and y.ndim==2:
            return 2*arcsin( np.sqrt(sin((x[1]-y[:,1])/2)**2 +
                             cos(x[1])*cos(y[:,1])*sin((x[0]-y[:,0])/2)**2) )
        elif x.ndim==2 and y.ndim==2:
            return 2*arcsin( np.sqrt(sin((x[:,1]-y[:,1])/2)**2 +
                             cos(x[:,1])*cos(y[:,1])*sin((x[:,0]-y[:,0])/2)**2) )
        return 2*arcsin( np.sqrt(sin((x[1]-y[1])/2)**2+cos(x[1])*cos(y[1])*sin((x[0]-y[0])/2)**2) )

    @staticmethod
    def fast_dist(x,y):
        """Fast inaccurate Euclidean distance squared calculation accounting for periodic
        boundary conditions in phi. This is not too bad near the equator.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        dvec : ndarray
        """

        # Account for discontinuity at phi=0 and phi=2*pi
        d = np.abs(x-y)
        ix = d[:,0]>pi
        d[ix,0] = pi-d[ix,0]%pi
        return ( d**2 ).sum(1)
    
    @classmethod
    def wrap_phi(cls, phi):
        if hasattr(phi, '__len__'):
            wrapix = phi>pi
            phi[wrapix] = phi[wrapix]%pi - pi
        return phi%pi - pi

    @classmethod
    def unwrap_phi(cls, phi):
        unwrapix = phi<0
        phi[unwrapix] += 2*pi
    
    @classmethod
    def unwrap_theta(cls, theta):
        theta[:] += pi/2
        theta[:] = theta%(2*pi)
        reverseix = theta>pi
        theta[reverseix] = 2*pi-theta[reverseix]
        theta[:] -= pi/2
        return reverseix

    def expand(self, factor, force=False, truncate_to_bounds=True):
        """Expand or contract grid by a constant factor. This operation maintains the
        center of mass projected onto the surface of the sphere fixed.

        Parameters
        ----------
        factor : float
        force : bool, False
            If True, carries out expansion even if points must be deleted.
        truncate_to_bounds : bool, True
            If True, then only keep points that fall within the bounds of the class.
        """

        samples = self.samples.copy()
        if not self.coarseGrid is None:
            coarseGrid = self.coarseGrid.copy()
        else:
            coarseGrid = None
        
        # To make discontinuities easier to work with, center everything about phi=0 where
        # phi in [-pi,pi] and theta=0 in [-pi/2,pi/2].
        assert factor>0
        try:
            # wrap sample phi's to [-pi,pi]
            if self.width[0]>self.width[1]:
                self.width = self.width[0]-2*pi, self.width[1]
                self.wrap_phi(samples[:,0])
                if not coarseGrid is None:
                    self.wrap_phi(coarseGrid[:,0])

            # check phi bounds
            if force and (np.ptp(mod_angle(samples[:,0]))*factor)>(2*pi):
                warn("Factor violates phi bounds.")
            else:
                assert (np.ptp(mod_angle(samples[:,0]))*factor)<=(2*pi)

            # check theta bounds
            if force and (np.ptp(samples[:,1])*factor)>pi:
                warn("Factor violates theta bounds.")
            else:
                assert ((np.ptp(samples[:,1])*factor)<=pi), "Factor violates theta bounds."
        except:
            # undo previous transformation if we must quit
            self.width = self.width[0]%(2*pi), self.width[1]
            raise Exception

        # center everything about (phi, theta) = (0,0)
        dangle = np.array([-np.mean(self.width), -np.mean(self.height)])[None,:]
        samples += dangle
        if not coarseGrid is None:
            coarseGrid += dangle
        
        # now run the expansion and contraction about the center of the tiling
        samples *= factor
        if not coarseGrid is None:
            coarseGrid *= factor

        # remove all sample points that wrap around sphere including both individual points that exceed
        # boundaries and all children of coarse grained points that exceed boundaries
        if truncate_to_bounds:
            width = self.width[0]+dangle[0,0], self.width[1]+dangle[0,0]
            height = self.height[0]+dangle[0,1], self.height[1]+dangle[0,1]
        else:
            width = -pi, pi
            height = -pi/2, pi/2
        samplesToRemove = np.where((samples[:,0]<width[0]) | (samples[:,0]>width[1]) |
                                   (samples[:,1]<height[0]) | (samples[:,1]>height[1]))[0].tolist()
        if not coarseGrid is None:
            coarseToRemove = np.where((coarseGrid[:,0]<width[0]) | (coarseGrid[:,0]>width[1]) |
                                      (coarseGrid[:,1]<height[0]) | (coarseGrid[:,1]>height[1]))[0].tolist()
        else:
            coarseToRemove = False
        
        #    # collect all children of coarse nodes to remove
        #    for ix in coarseToRemove:
        #        samplesToRemove += self.samplesByGrid[ix]
        #    samplesToRemove = list(set(samplesToRemove))  # remove duplicates
        # remove all samples
        if samplesToRemove:
            if self.iprint:
                print("Removing %d samples."%len(samplesToRemove))
            samples = np.delete(samples, samplesToRemove, axis=0)
            if samples.size==0:
                raise NoVoronoiTilesRemaining("No samples left in domain after expansion.")
        # remove all coarse grid centers
        if not coarseGrid is None and coarseToRemove:
            coarseGrid = np.delete(coarseGrid, coarseToRemove, axis=0)
            if len(coarseGrid)==0:
                coarseGrid = None
        
        # Undo earlier offsets
        samples -= dangle
        if not coarseGrid is None:
            coarseGrid -= dangle
        
        # By removing the offset of the COM, we might have angles that are beyond permissible limits. Put
        # theta back into [-pi/2,pi/2] and account for any reversals in phi if theta is outside that range
        # modulo
        reverseix = self.unwrap_theta(samples[:,1]) 
        # if theta is between [pi,2*pi] then phi must be flipped by pi
        samples[reverseix,0] += pi
        samples[:,0] = mod_angle(samples[:,0])
        if not coarseGrid is None:
            reverseix = self.unwrap_theta(coarseGrid[:,1]) 
            coarseGrid[reverseix,0] += pi
            coarseGrid[:,0] = mod_angle(coarseGrid[:,0])

        # put phi back in to [0,2*pi]
        if self.width[0]<0:
            self.width = self.width[0]+2*pi, self.width[1]
            # map phi back to [0,2*pi]
            self.unwrap_phi(samples[:,0])
            if not coarseGrid is None:
                self.unwrap_phi(coarseGrid[:,0])
        
        self.samples = samples
        if coarseGrid is None:
            self.delete_coarse_grid()
        else:
            self.set_coarse_grid(coarseGrid)
    
    def _default_plot_kw(self):
        return {'xlabel':r'$\phi$', 'ylabel':r'$\theta$', 'xlim':self.width,'ylim':self.height}

    def plot(self,
             fig=None,
             ax=None,
             kw_ax_set=None,
             apply_mod=False):
        """Plot coarse grid square plot of angles theta by phi.

        Parameters
        ----------
        fig : matplotlib.Figure, None
        ax : matplotlib.Axes, None
        kw_ax_set : dict, None
        apply_mod : bool, False
            If True, wrap phi to [-pi,pi].
        """

        if fig is None:
            fig, ax = plt.subplots()
        elif ax is None:
            ax = fig.add_subplot(1, 1, 1)

        for i in range(len(self.coarseGrid)):
            ix = self.samplesByGrid[i]
            if apply_mod:
                h = ax.plot(mod_angle(self.samples[ix,0]), self.samples[ix,1], 'o')[0]
                ax.plot(mod_angle(self.coarseGrid[i,0]), self.coarseGrid[i,1], 'x', c=h.get_mfc(), mew=3)
            else:
                h = ax.plot(self.samples[ix,0], self.samples[ix,1], 'o')[0]
                ax.plot(self.coarseGrid[i,0], self.coarseGrid[i,1], 'x', c=h.get_mfc(), mew=3)
        
        if kw_ax_set is None:
            kw_ax_set = self._default_plot_kw()
            if apply_mod:
                kw_ax_set['xlim'] = mod_angle(kw_ax_set['xlim'][0]), mod_angle(kw_ax_set['xlim'][1])

        ax.set(**kw_ax_set)
        return fig

    def plot_on_map(self,
                    fig=None,
                    ax=None,
                    fig_kw={'figsize':(5,5)},
                    ax_kw={},
                    plot_kw={'s':8, 'alpha':.2, 'color':'k','lw':0},
                    lon_offset=330):
        """Scatter plot on map of Earth.

        Parameters
        ----------
        fig : matplotlib.Figure, None
        ax : matplotlib.Axes, None
        fig_kw : dict, {'figsize':(5,5)}
        ax_kw : dict, {}
        plot_kw : dict, {'s':8, 'alpha':.2, 'color':'k','lw':0}
        lon_offset : float, 330
            To center voronoi grid above Africa.

        Returns
        -------
        matplotlib.Figure
        matplotlib.Axes
        """

        if fig is None:
            fig = plt.figure(**fig_kw)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        elif ax is None:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # add geographic features
        ax.add_feature(cfeature.LAND, zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=0)
        ax.add_feature(cfeature.BORDERS, zorder=0)
        ax.add_feature(cfeature.RIVERS, zorder=0)
        ax.add_feature(cfeature.LAKES, zorder=0)

        # show centers of voronoi cells
        ax.scatter(self.samples[:,0]/pi*180+lon_offset,
                   self.samples[:,1]/pi*180,
                   transform=ccrs.PlateCarree(),
                   zorder=1,
                   **plot_kw)
        return fig, ax 

    def pixelate(self, xy):
        """Assign given coordinates to pixel in self.samples.

        Parameters
        ----------
        xy : ndarray
            Pairs of coordinates in radians given as (phi, theta) equivalent to
            (longitude, latitude). The coordinates phi in [0,2*pi] and theta in
            [-pi/2,pi/2].

        Returns
        -------
        list of indices
            Pixel to which each coordinate belongs.
        """
        
        # check that lon and lat are within bounds used for this code
        assert ((xy[:,0]>=0) & (xy[:,0]<2*pi) & (-pi/2<=xy[:,1]) & (xy[:,1]<=pi/2)).all()
        
        # only pixelate unique coordinates so we don't have to waste time repeating distance calculation
        uxy, invix = np.unique(xy, axis=0, return_inverse=True)
        upixIx = self.closest_neighbor(uxy, ignore_zero=False)

        pixIx = [upixIx[i] for i in invix]
        return pixIx

    def within_limits(self, xy):
        """Check if given points are within the boundaries of this tesselation.
        
        Parameters
        ----------
        xy : ndarray
            Assuming that this is already within bounds phi in [0,2pi] and theta in
            [-pi/2,pi/2].

        Returns
        -------
        bool
        """

        assert type(xy) is np.ndarray
        if xy.ndim==1:
            # case where the interval includes the discontinuity at 2pi
            if self.width[0]>self.width[1]:
                return (((self.width[0]<=xy[0]) or (xy[0]<=self.width[1])) and
                        (self.height[0]<=xy[1]<=self.height[1]))
            return ((self.width[0]<=xy[0]<=self.width[1]) and
                    (self.height[0]<=xy[1]<=self.height[1]))
        
        assert xy.shape[1]==2, "Dimensions don't agree with angular coordinates."
        if self.width[0]>self.width[1]:
            return (((self.width[0]<=xy[:,0]) | (xy[:,0]<=self.width[1])) &
                    (self.height[0]<=xy[:,1]) & (xy[:,1]<=self.height[1])).all()
        return ((self.width[0]<=xy[:,0]) & (xy[:,0]<=self.width[1]) &
                (self.height[0]<=xy[:,1]) & (xy[:,1]<=self.height[1])).all()

    def slow_neighbor(self, xy):
        """Find closest neighbor by comparison with all pairwise distances. This is meant
        as a check for the fast algorithm.

        Parameters
        ----------
        xy : ndarray
            Only a single coordinate.

        Returns
        -------
        int
        """
        
        assert xy.ndim==1
        return np.argmin(self.dist(xy, self.samples))
#end PoissonDiscSphere
    
def cartesian_com(phi, theta):
    # center of mass calculated is to be calculated in 3D, so convert spherical
    # coordinates to Cartesian
    xyz = np.vstack((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))).T
    com = xyz.mean(0)
    com /= np.linalg.norm(com)

    # project Cartesian COM to spherical surface and expand samples and coarse grid points around that by
    # factor
    com = np.array([np.arctan2(com[1], com[0]), np.arccos(com[2])-pi/2])
    return com

@njit
def find_first_in_r(xy, xyOther, r):
    """Find index of first point that is within distance r. This avoid a distance
    calculation between all pairs if a faster condition can be satisfied.
    
    Parameters
    ----------
    xy : ndarray
        two elements
    xyOther : ndarray
        List of coordinates.
    r : float

    Returns
    -------
    int
        Index of either first element within r/2 or closest point if no point is
        within r/2.
    """
    
    dmin = 4  # knowing that max geodesic distance on spherical surface is pi
    minix = 0
    for i in range(len(xyOther)):
        d = 2*arcsin( np.sqrt(sin((xy[1]-xyOther[i,1])/2)**2 +
                              cos(xy[1])*cos(xyOther[i,1])*sin((xy[0]-xyOther[i,0])/2)**2) )
        # since closest possible spacing is r, a distance of r/2 indicates a guaranteed coarse neighbor
        if d<=(r/2):
            return i
        elif d<dmin:
            dmin = d
            minix = i
    return minix


class SphereCoordinate():
    """Coordinate on unit sphere. Contains methods for easy manipulation and translation
    of points. Sphere is normalized to unit sphere.
    """
    def __init__(self, *args, rng=None):
        """
        Parameters
        ----------
        (x,y,z) or vector or (phi,theta)
        rng : np.random.RandomState, None
        """

        self.update_xy(*args)
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
            
    def update_xy(self, *args):
        """Store both Cartesian and spherical representation of point."""

        if len(args)==2:  # assuming angles are given
            # theta is angle off z-axis
            # phi is that around x-y plane
            phi, theta = args
            assert 0<=phi<=(2*pi)
            assert 0<=theta<=pi
            self.vec = np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
            self.phi, self.theta = phi, theta
        else:  # assuming separate vector components are given
            assert len(args)==3 or len(args[0])==3
            if len(args)==3:
                self.vec = np.array(args)
            else:
                self.vec = args[0]
            
            # enforce unit normalization
            self.vec = self.vec / (np.nextafter(0,1) + np.linalg.norm(self.vec))
            self.phi, self.theta = self._vec_to_angle(*self.vec)
        self.angle = np.array([self.phi, self.theta])
    
    @classmethod
    def _angle_to_vec(cls, phi, theta):
        return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    
    @classmethod
    def _vec_to_angle(cls, x, y, z):
        if z<0:
            return arctan2(y, x)%(2*pi), arccos(max(z, -1))
        return arctan2(y, x)%(2*pi), arccos(min(z, 1))
           
    def random_shift(self,return_angle=True,bds=[0,1]):
        """Return a vector that is randomly shifted away from this coordinate. This is
        done by imagining that the north pole is aligned along this vector and then adding
        a random angle and then rotating the north pole to align with this vector.

        Angles are given relative to the north pole; that is, theta in [0,pi] and phi in
        [0,2*pi].

        Parameters
        ----------
        return_angle : bool,False
            If True, return random vector in form of a (phi,theta) pair.
        bds : tuple,[0,1]
            Bounds on uniform number generator to only sample between fixed limits of
            theta. This can be calculated using the formula
                (1+cos(theta)) / 2 = X
            where 0<=x<=1

        Returns
        -------
        randvec : ndarray
        """

        # setup rotation operation
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dphi, dtheta = (self.rng.uniform(0, 2*pi),
                        arccos(2*self.rng.uniform(*bds)-1))
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        if return_angle:
            newphi, newtheta=self._vec_to_angle( *randq.rotate(rotq.inv()).vec )
            if inSouthPole:
                # move back to south pole
                newtheta=pi-newtheta
            return newphi, newtheta
        newvec=randq.rotate(rotq.inv()).vec
        if inSouthPole:
            # move back to south pole
            newvec[-1]*=-1
        return newvec

    def rotate(self, rotvec, d):
        """
        Parameters
        ----------
        vec : ndarray
            Rotation axis
        d : float
            Rotation angle

        Returns
        -------
        SphereCoordinate
        """

        rotvec /= np.sqrt(rotvec[0]**2 + rotvec[1]**2 + rotvec[2]**2)
        a, b = cos(d/2), sin(d/2)
        rotq = Quaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        vec = self._angle_to_vec(self.phi, self.theta)
        vecq = Quaternion(0, vec[0], vec[1], vec[2])
        
        newvec = vecq.rotate(rotq).vec
        newphi, newtheta = self._vec_to_angle( newvec[0], newvec[1], newvec[2] )

        return SphereCoordinate(newphi%(2*pi), newtheta)

    def rotate_to_north_pole(self):
        """Parameters for rotating vector to north pole.

        To rotate back from north pole, just take the negative of the calculated angle.
        
        Returns
        -------
        ndarray
            Rotation axis.
        float
            Angle to rotate.
        """
        
        rotvec = np.cross( self.vec, np.array([0,0,1]) )
        rotvec /= np.linalg.norm(rotvec)
        d = arccos( self.vec[-1] )

        return rotvec, d

    def geo_dist(self, y):
        """Great circle distance to other point y.
        
        Parameters
        ----------
        y : SphereCoordinate or twople

        Returns
        -------
        float
        """
        
        if not isinstance(y, type(self)):
            if len(y)==2:
                return haversine([self.phi, self.theta], y)
            elif len(y)==3:
                return haversine((self.phi, self.theta),
                                 self._vec_to_angle(*y))
            raise Exception("Bad arg.")

        return haversine([self.phi, self.theta],
                         [y.phi, y.theta])

    def dot(self, y):
        if isinstance(y, type(self)):
            return self.vec.dot(y.vec)
        return self.vec.dot(y)

    def __repr__(self):
        coord = self.vec[0], self.vec[1], self.vec[2], self.phi, self.theta
        return "misc.globe.SphereCoordinate\nx=%1.4f, y=%1.4f, z=%1.4f\nphi=%1.4f, theta=%1.4f"%coord

    def __str__(self):
        coord = self.vec[0], self.vec[1], self.vec[2], self.phi, self.theta
        return "misc.globe.SphereCoordinate\nx=%1.4f, y=%1.4f, z=%1.4f\nphi=%1.4f, theta=%1.4f"%coord

    def __add__(self, y):
        assert isinstance(y, type(self))
        newvec = self.vec + y.vec
        if (newvec==0).all():
            raise Exception("Vectors are parallel. No well-defined average.")
        return SphereCoordinate(newvec)

    def __sub__(self, y):
        assert isinstance(y, type(self))
        newvec = self.vec - y.vec
        if (newvec==0).all():
            raise Exception("Vectors are parallel. No well-defined average.")
        return SphereCoordinate(newvec)

    def __lt__(self, y):
        assert isinstance(y, type(self))
        return self.vec < y.vec

    def __gt__(self, y):
        assert isinstance(y, type(self))
        return self.vec > y.vec

    def __eq__(self, y):
        assert isinstance(y, type(self))
        return np.array_equal(self.vec, y.vec)
#end SphereCoordinate


spec=[
       ('vec',float64[:]),
       ('phi',float64),
       ('theta',float64)
     ]
@jitclass(spec)
class jitSphereCoordinate():
    """Coordinate on a spherical surface. Contains methods for easy manipulation and
    translation of points. Sphere is normalized to unit sphere.

    This is a slimmed down version of SphereCoordinate.

    theta in [0, 2*pi]
    phi in [0, pi]
    """

    def __init__(self, phi, theta):
        """
        Parameters
        ----------
        phi : float
        theta : float
        """
                
        self.update_xy(phi, theta)
            
    def update_xy(self, phi, theta):
        assert 0<=phi<=(2*pi)
        assert 0<=theta<=pi
        self.vec = np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
        self.phi, self.theta = phi, theta
    
    def _angle_to_vec(self,phi,theta):
        return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    
    def _vec_to_angle(self, x, y, z):
        if z<0:
            return arctan2(y,x)%(2*pi), arccos(max(z, -1))
        return arctan2(y,x)%(2*pi), arccos(min(z, 1))

    def random_shift(self, bds):
        """Return a vector that is randomly shifted away from this coordinate. This is done by
        imagining that the north pole is aligned along this vector and then adding a random angle
        and then rotating the north pole to align with this vector.

        Angles are given relative to the north pole; that is, theta in [0,pi] and phi in [0,2*pi].

        Parameters
        ----------
        bds : tuple, [0,1]
            Bounds on uniform number generator to only sample between fixed limits of theta. This
            can be calculated using the formula
                (1+cos(theta)) / 2 = X
            where 0<=x<=1

        Returns
        -------
        newphi : float
        newtheta : float
        """

        # setup rotation operation
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dphi=np.random.uniform(0, 2*pi)
        dtheta=arccos(2*np.random.uniform(bds[0], bds[1])-1)
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        vec=randq.rotate(rotq.inv()).vec
        newphi, newtheta=self._vec_to_angle( vec[0], vec[1], vec[2] )
        if inSouthPole:
            # move back to south pole
            newtheta=pi-newtheta
        return newphi, newtheta
          
    def random_shift_controlled(self, bds, r1, r2):
        """Same as random_shift() except with explicit control of random numbers by passing them in.
        """

        # setup rotation operation
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dphi = r1*2*pi  #np.random.uniform(0, 2*pi)
        dtheta = arccos(2*(r2*(bds[1]-bds[0])+bds[0])-1)  #arccos(2*np.random.uniform(bds[0], bds[1])-1)
        dvec = self._angle_to_vec(dphi, dtheta)
        randq = jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        vec = randq.rotate(rotq.inv()).vec
        newphi, newtheta = self._vec_to_angle( vec[0], vec[1], vec[2] )
        if inSouthPole:
            # move back to south pole
            newtheta = pi-newtheta
        return newphi, newtheta

    def rotate(self, rotvec, d):
        """
        Parameters
        ----------
        vec : ndarray
            Rotation axis
        d : float
            Rotation angle

        Returns
        -------
        newphi : float
        newtheta : float
        """

        rotvec=rotvec/np.sqrt(rotvec[0]**2 + rotvec[1]**2 + rotvec[2]**2)
        a, b=cos(d/2), sin(d/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        vec=self._angle_to_vec(self.phi, self.theta)
        vecq=jitQuaternion(0, vec[0], vec[1], vec[2])
        
        newvec=vecq.rotate(rotq).vec
        newphi, newtheta=self._vec_to_angle( newvec[0], newvec[1], newvec[2] )

        return jitSphereCoordinate(newphi%(2*pi), newtheta)

    def shift(self, dphi, dtheta):
        """
        Return a vector that is randomly shifted away from this coordinate. This is done by
        imagining that hte north pole is aligned along this vector and then adding a random angle
        and then rotating the north pole to align with this vector.

        Angles are given relative to the north pole; that is, theta in [0,pi] and phi in [0,2*pi].

        Parameters
        ----------
        dphi : float
            Change to azimuthal angle.
        dtheta : float
            Change to polar angle.

        Returns
        -------
        newphi : float
        newtheta : float
        """
        raise NotImplementedError
        # setup rotation operation
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        vec=randq.rotate(rotq.inv()).vec
        newphi, newtheta=self._vec_to_angle( vec[0], vec[1], vec[2] )
        if inSouthPole:
            # move back to south pole
            newtheta=pi-newtheta
        return jitSphereCoordinate(newphi%(2*pi), self.unwrap_theta(newtheta))

    def unwrap_theta(self, theta):
        theta=theta%(2*pi)
        if theta>pi:
            return pi-theta%pi
        return theta
#end jitSphereCoordinate


spec=[
        ('real',float64),
        ('vec',float64[:])
    ]
@jitclass(spec)
class jitQuaternion():
    """Faster quaternion class with limited functionality.
    """
    def __init__(self,a,b,c,d):
        a*=1.
        b*=1.
        c*=1.
        d*=1.
        self.real=a
        self.vec=np.array([b,c,d])
        
    def inv(self):
        negvec=-self.vec
        return jitQuaternion(self.real, negvec[0], negvec[1], negvec[2])
    
    def hprod(self,t):
        """Right side Hamiltonian product."""
        p=[self.real, self.vec[0], self.vec[1], self.vec[2]]
        t=[t.real, t.vec[0], t.vec[1], t.vec[2]]
        """Hamiltonian product between two quaternions."""
        return jitQuaternion( p[0]*t[0] -p[1]*t[1] -p[2]*t[2] -p[3]*t[3],
                              p[0]*t[1] +p[1]*t[0] +p[2]*t[3] -p[3]*t[2],
                              p[0]*t[2] -p[1]*t[3] +p[2]*t[0] +p[3]*t[1],
                              p[0]*t[3] +p[1]*t[2] -p[2]*t[1] +p[3]*t[0] )
    
    def rotmat(self):
        qr=self.real
        qi, qj, qk=self.vec
        return np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
                         [2*(qi*qj+qk*qr), 1-2*(qi**2+qk**2), 2*(qj*qk-qi*qr)],
                         [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]])

    def rotate(self,r):
        """Rotate this quaternion by the rotation specified in the given quaternion. The
        rotation quaternion must be of form cos(theta/2) + (a i, b j, c k)*sin(theta/2)

        Parameters
        ----------
        r : Quaternion
        """
        return r.hprod( self.hprod( r.inv() ) )

    def __str__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])
    
    def __repr__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])
#end jitQuaternion


class Quaternion():
    """Basic quaternion class. This can be used to represent vectors and efficient
    rotation operations on them.
    """
    def __init__(self, a, b, c, d):
        self.real = a  # magnitude of vector
        self.vec = np.array([b,c,d])  # normalized components of vector
        
    def inv(self):
        negvec = -self.vec
        return Quaternion(self.real, *negvec)
    
    def hprod(self,t):
        """Right side Hamiltonian product.
        """

        p = [self.real] + self.vec.tolist()
        t = [t.real] + t.vec.tolist()

        # Hamiltonian product between two quaternions
        return Quaternion( p[0]*t[0] -p[1]*t[1] -p[2]*t[2] -p[3]*t[3],
                           p[0]*t[1] +p[1]*t[0] +p[2]*t[3] -p[3]*t[2],
                           p[0]*t[2] -p[1]*t[3] +p[2]*t[0] +p[3]*t[1],
                           p[0]*t[3] +p[1]*t[2] -p[2]*t[1] +p[3]*t[0] )
    
    def rotmat(self):
        qr = self.real
        qi, qj, qk = self.vec
        return np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
                         [2*(qi*qj+qk*qr), 1-2*(qi**2+qk**2), 2*(qj*qk-qi*qr)],
                         [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]])

    def rotate(self, r):
        """Rotate this quaternion by the rotation specified in the given quaternion. The
        rotation quaternion must be of form cos(theta/2) + (a i, b j, c k)*sin(theta/2)

        Parameters
        ----------
        r : Quaternion
        """

        return r.hprod( self.hprod( r.inv() ) )

    def __str__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])
    
    def __repr__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])

    def __eq__(self,y):
        if not type(y) is Quaternion:
            raise NotImplementedError

        if y.real==self.real and np.array_equal(y.vec,self.vec):
            return True
        return False
#end Quaternion



class GreatCircle():
    """Great circle that lives on unit sphere in 3D. This keeps track of it by using the
    orthogonal plane and defining an arbitrary starting point for generating the full ring
    of points.
    """
    def __init__(self, w, startvec=None):
        """
        Parameters
        ----------
        w : ndarray
            Vector normal to defining plane.
        startvec : ndarray, None
            Always start tracing out great circle from this point. If not specified, try
            to calculate some arbitrary starting point.
        """
        
        assert w.size==3
        
        self.w = w / np.linalg.norm(w)
        if startvec is None:
            # choose arbitrary point along great circle as a starting point for cycling
            randvec = np.random.normal(size=3)
            self.startvec = np.cross(randvec, w)
            self.startvec /= np.linalg.norm(self.startvec)
        else:
            assert startvec.size==3
            self.startvec = startvec
            assert np.isclose(self.startvec.dot(w), 0)
        
    def ring(self, as_angle=False):
        """Define a function to trace out a great circle given a vector normal to its
        plane and parameterized by the angle of rotation around the circle.

        Parameters
        ----------
        as_angle : bool, False
            If True, returned function gives ring in terms of angular coordinates.
        
        Returns
        -------
        function
        """
        
        G0 = SphereCoordinate(self.startvec)
        
        if as_angle:
            def ring_fcn(omega, G0=G0, w=self.w):
                result = G0.rotate(w, omega)
                return result.phi, result.theta
            return ring_fcn
        
        def ring_fcn(omega, G0=G0, w=self.w):
            result = G0.rotate(w, omega)
            return result.vec
        return ring_fcn

    def intersect(self, v):
        """Find the intersections of two great circles using their normal vectors. This
        assumes that the vectors are normalized.

        Parameters
        ----------
        v : ndarray or GreatCircle

        Returns
        -------
        ndarray
            Two solutions as xyz coordinations.
        """
        
        if isinstance(v, type(self)) or 'w' in v.__dict__.keys():
            v = v.w
        else:
            assert np.linalg.norm(v)==1 and v.size==3
        w = self.w
        if np.array_equal(v, w):
            raise Exception("Great circles are the same.")
        
        # need a way of keeping track of whether or not the following calculate converges
        fancyD = (v[0]*w[2] - w[0]*v[2]) / (w[1]*v[2] - v[1]*w[2])
        xyz = np.zeros(3)
        xyz[0] = np.sqrt(w[2]**2 / ((1+fancyD**2)*w[2]**2 + (w[0]+w[1]*fancyD)**2))
        xyz[1] = np.sqrt(fancyD**2 * xyz[0]**2)
        # being careful for precision errors that could drive this to be negative
        # just set this to 0 if that is the case and accept that x and y coordinates will
        # be slightly wrong
        z2 = 1-xyz[0]**2-xyz[1]**2
        if z2>0:
            xyz[2] = np.sqrt(z2)
        
        # meet condition for plane passing through origin
        if np.isclose(w.dot(xyz), 0):  #+++
            signsAreCorrect = True
        else:
            signsAreCorrect = False
            
        if not signsAreCorrect:  #-++
            xyz[0] *= -1
            if np.isclose(w.dot(xyz), 0):
                signsAreCorrect = True
        if not signsAreCorrect:  #+-+
            xyz[0] *= -1
            xyz[1] *= -1
            if np.isclose(w.dot(xyz), 0):
                signsAreCorrect = True
        if not signsAreCorrect:  #--+
            xyz[0] *= -1
            if np.isclose(w.dot(xyz), 0):
                signsAreCorrect = True
        assert signsAreCorrect

        return np.vstack((xyz, -xyz))

    @classmethod
    def bisector(cls, x, y):
        """Given two points on the sphere, return an equation for the great circle passing
        between the two equidistantly. Great circle is oriented towards y.

        Parameters
        ----------
        x : SphereCoordinate
        y : SphereCoordinate

        Returns
        -------
        function
            Parameterized by angle theta where theta=0 is the point that is halfway
            between x and y on the sphere.
            Plane is oriented towards y from x.
        """

        assert y!=x
        w = y.vec - x.vec  # rotation axis (normal to plane of great circle)
        G0 = x + y  # midpoint vector that points to one point along great circle

        return GreatCircle(w, startvec=G0.vec)
    
    @classmethod
    def ortho(cls, v, w0):
        """Define great circle that passes thru w0 and forms a plane orthogonal to the
        geodesic from v to w0.
        
        Plane is oriented such that normal vector points towards v.
        
        Parameters
        ----------
        v : SphereCoordinate
            Start of geodesic.
        w0 : SphereCoordinate
            Point on great circle to be of minimal distance from v.
            
        Returns
        -------
        GreatCircle
        """
        
        # construct normal vector for defining great circle
        comp1 = v.vec.dot(w0.vec) * w0.vec
        w = v.vec - comp1
        
        return GreatCircle(w, startvec=w0.vec)

    def __str__(self):
        return f"misc.globe.GreatCircle: omega={self.w}"

    def __repr__(self):
        return f"misc.globe.GreatCircle: omega={self.w}"
#end GreatCircle



class GreatCircleIntersect():
    """Intersection of two great circles.
    """
    def __init__(self, g1, g2, xyz, d):
        """
        Parameters
        ----------
        g1 : GreatCircle
        g2 : GreatCircle
        xyz : ndarray
            Point of intersection.
        d : float
            Distance from center point bounded by great circles.
        """
        
        self.g1 = g1
        self.g2 = g2
        self.xyz = xyz
        self.d = d
        
    def __eq__(self, y):
        return ((((np.array_equal(self.g1, y.g1) and np.array_equal(self.g2, y.g2)) or
                  (np.array_equal(self.g1, y.g2) and np.array_equal(self.g2, y.g1)))) and
                 np.array_equal(self.xyz, y.xyz))

    def __str__(self):
        return f"misc.globe.GreatCircleIntersect\n{self.g1}\n{self.g2}"

    def __repr__(self):
        return f"misc.globe.GreatCircleIntersect\n{self.g1}\n{self.g2}"
#end GreatCircleIntersect



class VoronoiCell():
    """Voronoi cell defined by the set of vertices denoting intersections of edges.

    This can be used to determine cell boundaries. There are many places this could be
    sped up including with the insertion and consideration of new edges.

    Note that terminology "edges", "boundaries", "cuts", and "facets" are used
    interchangeably.
    """
    def __init__(self, center, rng=None):
        """
        Parameters
        ----------
        center : SphereCoordinate
            Center of cell.
        rng : np.random.RandomState, None
        """

        self.center = center
        self.vertices = []
        self.edges = []
        self.rng = rng or np.random

        # Generate a plane passing thru center and tangential to sphere. This will be used
        # to determine angular coordinates about the center.
        self.x = np.cross(self.rng.normal(size=3), center.vec)
        self.x /= np.linalg.norm(self.x)
        self.y = np.cross(self.x, -center.vec)

    def lip(self, pts):
        """Find lip bounding the center with closest two points.

        Parameters
        ----------
        pts : list of SphereCoordinate

        Returns
        -------
        list of ints
            Indices of the points that are used to determine bisecting great circles that
            define lips boundaries.
        """

        # find lip wrapping center starting with pair of closests points to center
        d = np.array([self.center.geo_dist(p) for p in pts])
        assert (d>0).all(), "Center point shouldn't be included."
        closeptsIx = np.argsort(d)[:2].tolist()
        closepts = pts[closeptsIx[0]], pts[closeptsIx[1]]

        # find intersections of the encompassing two geodesics
        lipEdges = [GreatCircle.bisector(p, self.center) for p in closepts]
        vertices = lipEdges[0].intersect(lipEdges[1])

        return vertices, lipEdges

    def initialize_with_tri(self, pts):
        """Initialize cell with a triangle generated from closest possible set of points,
        which may indeed be the three closest points but not necessarily.

        Parameters
        ----------
        pts : list of SphereCoordinate

        Returns
        -------
        list of ints
            Indices of the points that are used to determine bisecting great circles that
            define triangle boundaries.
        """

        # find lip wrapping center starting with pair of closests points to center
        d = np.array([self.center.geo_dist(p) for p in pts])
        assert (d>0).all(), "Center point shouldn't be included."
        closeptsIx = np.argsort(d)[:2].tolist()
        closepts = pts[closeptsIx[0]], pts[closeptsIx[1]]

        # find intersections of the encompassing two geodesics
        lipEdges = [GreatCircle.bisector(p, self.center) for p in closepts]
        vertices = lipEdges[0].intersect(lipEdges[1])
        
        # determine third edge to cut out one vertex (this is arbitrary)
        try:
            thisV = vertices[0]
            thisVix = 0
            sortix, checkResult = self._third_edge(thisV, pts, closeptsIx)
        except AssertionError:  # try other vertex
            thisV = vertices[1]
            thisVix = 1
            sortix, checkResult = self._third_edge(thisV, pts, closeptsIx)
        
        # build lip cell as a starting point
        # note that direction of rotation about lip is arbitrary, and you could check this
        # be reversing the order of defn
        v1, v2 = SphereCoordinate(vertices[0]), SphereCoordinate(vertices[1])
        self.edges = [(v1, v2, lipEdges[0]),
                      (v1, v2, lipEdges[1])]
        self.vertices = [v1, v2]
        self.add_cut(GreatCircle.bisector(pts[sortix[0]], self.center))

        return sorted(closeptsIx + [sortix[0]])
    
    def _third_edge(self, thisV, pts, closeptsIx):
        # check for any points that are on the same side as thisV and are close enough
        posPlane = GreatCircle.ortho(SphereCoordinate(thisV), self.center)
        twiced = self.center.geo_dist(thisV) * 2
        checkResult = [self._check_pt(pt, posPlane, twiced) for pt in pts]
        checkResult[closeptsIx[0]] = np.inf
        checkResult[closeptsIx[1]] = np.inf
        assert any(np.isfinite(checkResult))

        # sort remaining points by distance
        sortix = np.argsort(checkResult)
        return sortix, checkResult

    def _check_pt(self, pt, posPlane, d):
        """Return distance to pt if it is on positive side of plane and within distance of center.
        """
        
        if pt.dot(posPlane.w)<0:
            return np.inf
        
        thisd = pt.geo_dist(self.center)
        if thisd < d:
            return thisd
        return np.inf

    def add_edge(self, p1, p2, G=None):
        """Add edge by specifying the two vertices. These should be ordered such that the
        cross product p1xp2 points into the cell. Otherwise, this can be done
        automatically by specifying the center.
        
        Parameters
        ----------
        p1 : SphereCoordinate
        p2 : SphereCoordinate
        G : GreatCircle, None
            Option to specify the great circle through which the edge passes.

        Returns
        -------
        bool
        """
        
        if isinstance(p1, np.ndarray):
            p1 = SphereCoordinate(p1)
        if isinstance(p2, np.ndarray):
            p2 = SphereCoordinate(p2)
        
        if G is None:
            w = np.cross(p1.vec, p2.vec)
            if self.center.dot(w)<0:
                temp = p1
                p1 = p2
                p2 = temp

            if np.array_equal(p1.vec, -p2.vec):
                raise Exception("Hemispheric edges not allowed.")

            newEdge = (p1, p2, GreatCircle(np.cross(p1.vec, p2.vec)))
        else:
            # could insert check here to make sure that the order of p1 and p2 is
            # consistent with the orientation of the GreatCircle
            newEdge = (p1, p2, G)

        # first check that new edge isn't extraneous
        if len(self.edges)>2:
            if not (self.inside(newEdge[0]) and self.inside(newEdge[1])):
                return False

        self.add_vertex(p1)
        self.add_vertex(p2)
        self.edges.append(newEdge)
        return True
    
    def add_vertex(self, newv):
        """Add vertex that isn't already in list.

        Parameters
        ----------
        newv : SphereCoordinate
        """

        if all([v!=newv for v in self.vertices]):
            self.vertices.append(newv)

    def order_vertices(self):
        """Order vertices in a counterclockwise fashion about the center.
        """
        
        self.angle = np.zeros(len(self.vertices))
        for i in range(self.angle.size):
            self.angle[i] = arctan2(self.vertices[i].dot(self.y),
                                    self.vertices[i].dot(self.x))
        sortix = np.argsort(self.angle)
        self.angle = self.angle[sortix]
        self.vertices = [self.vertices[i] for i in sortix]

    def add_cut(self, G):
        """Determine new edge that should be added to cell given a cut of the cell with a
        great circle.

        There are several cases that give poor cuts that are avoided below.
            1. No intersections.
            2. Intersection at one vertex, which doesn't add a new facet.
            3. Intersection at two vertices, which doesn't add a new facet. This could
               only be the case if the shape to be cut is a lip.

        Parameters
        ----------
        G : GreatCircle

        REturns
        -------
        bool
            Returns False if unsuccessful cut.
        """

        assert len(self.edges)>1
        
        # there are at most two points of intersection
        # first get all points of intersection with existing edges to find ones that are
        # on the boundaries of the existing cell
        xyz = np.vstack([G.intersect(edge[2]) for edge in self.edges])
        xyz = [xyz_ for xyz_ in xyz if self.inside(xyz_)]
        # if new cut goes through an existing vertex, there will seemingly be more than
        # two intersections, so we must count the number of unique vertices carefully by
        # accounting for precision errors
        if len(xyz)>=2:
            uxyz, uix = np.unique(xyz, axis=0, return_inverse=True)
            if uxyz.shape[0]>1:
                uxyz, newuix = self._combine_close_rows(uxyz)
                uix = uix[newuix]
            uix = uix.tolist()
            assert len(uxyz)<=2, uxyz  # there is a persistent precision issue if this is
                                       # not satisfied b/c there could only be two
                                       # intersections
            if len(uxyz)==1 or np.array_equal(uxyz[0], -uxyz[1]): return False
            xyz = [uxyz[0], uxyz[1]]

        # if zero points of intersection or one point of intersection at an already
        # existing vertex, then disregard
        if len(xyz)<=1: return False
        
        self.add_edge(xyz[0], xyz[1])  # will be auto-oriented towards center
        
        # now eliminate extraneous vertices by accounting for this new edge
        goodVertices = []
        for v in self.vertices: 
            if self.inside(v):
                goodVertices.append(v)
        self.vertices = goodVertices
        self.reconstruct_hull()
        return True
    
    def reconstruct_hull(self):
        """Reconstruct hull by looping around outside."""

        self.order_vertices()
        self.edges = []
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i+1)%len(self.vertices)]
            G = GreatCircle(np.cross(p1.vec, p2.vec))
            self.edges.append((p1, p2, G))

    def inside(self, p, tol=1e-8, detailed=False):
        """Check if given point is inside or outside Voronoi cell.

        TODO: Faster loop method would stop loop immediately after detecting at least one
        outside point.
        
        Parameters
        ----------
        p : SphereCoordinate
        tol : float, 1e-8
            Allow for potentially this much precision error for calculating whether or not
            boundary condition is positive.
            This needs to be small enough that the distance check used in .add_cut() won't
            have issues distinguishing true intersections from seemings ones that are just
            close to the boundary. I think a good guide is that this should be smaller
            than the global precision tol used for that.
        detailed : bool, False
            If True, return full list of edge-by-edge comparison.
        
        Returns
        -------
        bool (or list of bool)
            True if inside.
        """
        
        if detailed:
            return [(p.dot(edge[2].w) + tol)>0 for edge in self.edges]
        return all([(p.dot(edge[2].w) + tol)>0 for edge in self.edges])
    
    def outside(self, p):
        return (not self.inside(p))

    def boundaries(self, stepsize=1e-2, as_angle=False):
        """Return list of points outlining boundaries of the cell.

        Parameters
        ----------
        stepsize : float, 1e-2
        as_angle : bool, False
        """

        xyz = []
        for i in range(len(self.edges)):
            p1, p2, thisEdge = self.edges[i]

            x = thisEdge.startvec
            y = -np.cross(x, thisEdge.w)
            
            theta1 = arctan2(p1.dot(y), p1.dot(x))
            theta2 = arctan2(p2.dot(y), p2.dot(x))

            # if this passes thru discontinuity must account for jump
            # remember that edges are defined such that the normal vector points towards
            # the center
            if (theta2-theta1)>pi:
                theta1 += 2*pi
            elif (theta2-theta1)<-pi:
                theta1 -= 2*pi
            #assert theta2 > theta1

            f = thisEdge.ring(as_angle=as_angle)
            theta = np.arange(int((theta2-theta1)/stepsize)+1) * stepsize + theta1
            xyz.append(np.vstack([f(i) for i in theta]))

        return xyz

    @staticmethod
    def _combine_close_rows(X, tol=PRECISION):
        """Combine rows that are within tolerance.
        """
        
        groups = []
        d = squareform(pdist(X))

        remainingix = list(range(len(X)))
        while remainingix:
            thisix = remainingix.pop(0)
            groups.append([thisix])

            ix = np.where((0<d[thisix])&(d[thisix]<tol))[0].tolist()
            for ix_ in ix:
                # we are going to assume that the groups are obviously identifiable,
                # such that we don't need to make sure we haven't already counted this
                # in another group of points that's further away from this one but closer
                # to points in between
                groups[-1].append(remainingix.pop(remainingix.index(ix_)))
        
        newCenters = np.zeros((len(groups), 3))
        newix = np.zeros(len(groups), dtype=int)
        for i, g in enumerate(groups):
            newCenters[i] = X[g].mean(0)
            newix[i] = g[0]
        return newCenters, newix
#end VoronoiCell



# ================= #
# Exception classes #
# ================= #
class NoVoronoiTilesRemaining(Exception):
    pass
