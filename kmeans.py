#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pyspark import SparkContext



from math import degrees, radians, sin, cos, sqrt, asin, atan2

# Radius of the Earth, in km
EARTH_RADIUS = 6371

class LatLonPoint:
    def __init__(self, lat, lon):
        self.lat = float(lat)
        self.lon = float(lon)

class CartesianPoint:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

# Convert (lat, lon) to (x, y, z)
def latLonToCartesian(point):
    latRad = radians(point.lat)
    lonRad = radians(point.lon)
    
    x = EARTH_RADIUS * cos(latRad) * cos(lonRad)
    y = EARTH_RADIUS * cos(latRad) * sin(lonRad)
    z = EARTH_RADIUS * sin(latRad)
    
    return CartesianPoint(x, y, z)

# Convert (x, y, z) to (lat, lon)
def cartesianToLatLon(point):
    lat = degrees(asin(point.z / EARTH_RADIUS))
    lon = degrees(atan2(point.y, point.x))
    
    return LatLonPoint(lat, lon)

# Based on distanceMeasure, calculate the distances from a point to the centroids
# Return the index of the centroid closest to the point
def closestPoint(point, centroids, distanceMeasure):
    minDist = float("inf")
    closest = -1
    for idx, centroid in centroids.items():
        if distanceMeasure == "greatcircle":
            dist = greatCircleDistance(point, centroid)
        else:
            dist = euclideanDistance(point, centroid)
        if dist < minDist:
            minDist = dist
            closest = idx
            
    return closest

# Add two CartesianPoints
def addPoints(a, b):
    return CartesianPoint(a.x + b.x, a.y + b.y, a.z + b.z)

# Return a new centroid by calculating the mean of points in the cluster
# sumPoints: CartesianPoint representing the sum of the points in the cluster
# numPoints: number of points in the cluster
def dividePoints(sumPoints, numPoints):
    result = CartesianPoint(sumPoints.x / numPoints, sumPoints.y / numPoints, sumPoints.z / numPoints)
    return cartesianToLatLon(result)

# Return the Euclidean distance between two LatLonPoints
def euclideanDistance(a, b):
    aCart = latLonToCartesian(a)
    bCart = latLonToCartesian(b)
    
    return sqrt((aCart.x - bCart.x)**2 + (aCart.y - bCart.y)**2 + (aCart.z - bCart.z)**2)

# Return the great circle distance between two LatLonPoints
def greatCircleDistance(p1, p2):
    lat_diff = radians(p2.lat) - radians(p1.lat)
    lon_diff = radians(p2.lon) - radians(p1.lon)    
    
    a = (sin(lat_diff/2))**2 + cos(radians(p1.lat)) * cos(radians(p2.lat)) * (sin(lon_diff/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return EARTH_RADIUS * c




if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    # convergeDist is set to 0.1 as stated in the project guidelines
    convergeDist = float(0.1)
    iterationDist = float("inf")

    # Validate command line arguments
    if len(sys.argv) != 5:
        print ("usage: python " + sys.argv[0] + " <input_path> <output_path> <distance_measure> <k>\n")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    distanceMeasure = sys.argv[3].lower()
    if distanceMeasure != "euclidean" and distanceMeasure != "greatcircle":
        print ("distance_measure must be either \"Euclidean\" or \"GreatCircle\"\n")
        sys.exit(1)

    k = int(sys.argv[4])
    if k < 1:
        print ("k should be larger than 0\n")
        sys.exit(1)


    # Parse data
    filtered_data = sc.textFile("file:/" + input_file_path)
    latlon = filtered_data.map(lambda x: x.split(","))     .map(lambda x: [LatLonPoint(x[0], x[1]), x[2:]]).cache()

    # Use a sample of points as the initial centroids
    centroids = latlon.map(lambda x: x[0]).takeSample(False, k, 1)
    centroids = dict(zip(range(0, k), centroids))


    while iterationDist > convergeDist:
        # Map each point to a key-value pair, where the key is the closest centroid
        closest = latlon.map(lambda point: (closestPoint(point[0], centroids, distanceMeasure), point))

        # Count the number of points in each centroid
        numPoints = closest.countByKey()

        # Calculate the new centroids using the sum of the points and the number of points in the centroid
        newMeans = closest.map(lambda closest, point: (closest, latLonToCartesian(point[0])))         .reduceByKey(addPoints)         .map(lambda centroid, sumPoints: (centroid, dividePoints(sumPoints, numPoints[centroid])))

        # Calculate the distances between the old and new centroids
        if distanceMeasure == "greatcircle":
            distances = newMeans.map(lambda idx, mean: greatCircleDistance(centroids[idx], mean))
        else:
            distances = newMeans.map(lambda idx, mean: euclideanDistance(centroids[idx], mean))

        # Use the sum of the distances between old and new centroids as the convergeDist measure
        iterationDist = distances.sum()

        # Update centroids to the new means
        centroids = newMeans.collectAsMap()


    # Save the list of points and their clusters to a file
    closest.map(lambda closest, point: str(closest) + "," + str(point[0].lat) + "," + str(point[0].lon) + "," + ",".join(s.encode("utf-8","ignore") for s in point[1]))     .saveAsTextFile("file:/" + output_file_path)

    # Print the final centroids
    for idx, centroid in centroids.items():
        print(str(idx) + "," + str(centroid.lat) + "," + str(centroid.lon))

    # Calculate the mean distance between points and their nearest centroid
    if distanceMeasure == "greatcircle":
        distsFromCentroid = closest.map(lambda closest, point: greatCircleDistance(centroids[closest], point[0]))
    else:
        distsFromCentroid = closest.map(lambda closest, point: euclideanDistance(centroids[closest], point[0]))
    print(distsFromCentroid.mean())


# In[ ]:




