import skimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy
matplotlib.use('Qt5Agg')  

image = skimage.io.imread("data/campus.tiff")
# print("successfully loaded image")


# print(np.shape(image))
# print(image.dtype)
# print(bin(image[100, 100]))

image = image.astype(np.float64)

#########Linearization
# black is 150 -> , white is 4095 -> 1
image = image - 150.0
image = image / (4095.0-150.0)
image = np.clip(image, 0.0, 1.0)

# fig,axs = plt.subplots(2, 2, figsize=(12, 4))
# axs[0,0].imshow(image[::2, ::2], cmap='gray')
# axs[0,1].imshow(image[::2, 1::2], cmap='gray')
# axs[1,0].imshow(image[1::2, ::2], cmap='gray')
# axs[1,1].imshow(image[1::2, 1::2], cmap='gray')
# plt.show()

###############White Balancing (Automatic)
reds = image[::2, ::2]
greens1 = image[::2, 1::2]
greens2 = image[1::2, ::2]
blues = image[1::2, 1::2]


rmax = np.max(reds)
gmax = np.max((np.max(greens1), np.max(greens2)))
bmax = np.max(blues)

WB_white_world = np.empty(image.shape, dtype=image.dtype)
WB_white_world[::2, ::2] = reds*gmax/rmax
WB_white_world[::2, 1::2] = greens1
WB_white_world[1::2, ::2] = greens2
WB_white_world[1::2, 1::2] = blues*gmax/bmax

ravg = np.average(reds)
gavg = np.average((np.average(greens1), np.average(greens2)))
bavg = np.average(blues)
WB_gray_world = np.empty(image.shape, dtype=image.dtype)
WB_gray_world[::2, ::2] = reds*gavg/ravg
WB_gray_world[::2, 1::2] = greens1
WB_gray_world[1::2, ::2] = greens2
WB_gray_world[1::2, 1::2] = blues*gavg/bavg
#print(ravg, bavg, gavg)

WB_camera = np.empty(image.shape, dtype=image.dtype)
WB_camera[::2, ::2] = reds*2.394531
WB_camera[::2, 1::2] = greens1
WB_camera[1::2, ::2] = greens2
WB_camera[1::2, 1::2] = blues*1.597656

# fig,axs = plt.subplots(2, 2, figsize=(12, 4))
# axs[0,0].imshow(WB_white_world, cmap='gray')
# axs[0,1].imshow(WB_gray_world, cmap='gray')
# axs[1,0].imshow(WB_camera, cmap='gray')


#########Use this to select an automatic white balancing method
WB_choice=WB_camera

#############Demosiacing
rgb_out = np.zeros((4016, 6016, 3), dtype=image.dtype)

grid_y = np.arange(4016)
grid_x = np.arange(6016)
target_points = np.vstack(np.meshgrid(grid_y, grid_x, indexing='ij')).reshape(2, -1).T

red_interp = scipy.interpolate.RegularGridInterpolator((grid_y[::2], grid_x[::2]), 
                WB_choice[::2,::2], method="linear", 
                bounds_error=False, fill_value=None)
rgb_out[:, :, 0] = red_interp(target_points).reshape(4016, 6016)


blue_interp = scipy.interpolate.RegularGridInterpolator((grid_y[1::2], grid_x[1::2]), 
                WB_choice[1::2,1::2], method="linear", 
                bounds_error=False, fill_value=None)
rgb_out[:, :, 2] = blue_interp(target_points).reshape(4016, 6016)

green_interp1 = scipy.interpolate.RegularGridInterpolator((grid_y[::2], grid_x[1::2]), 
                WB_choice[::2,1::2], method="linear", 
                bounds_error=False, fill_value=None)
green_interp2 = scipy.interpolate.RegularGridInterpolator((grid_y[1::2], grid_x[::2]), 
                WB_choice[1::2,0::2], method="linear", 
                bounds_error=False, fill_value=None)
green_interp_final = (green_interp1(target_points)+green_interp2(target_points))/2
green_interp_final = green_interp_final.reshape(4016, 6016)
green_interp_final[::2, 1::2] = WB_choice[::2, 1::2]
green_interp_final[1::2, 0::2] = WB_choice[1::2, ::2]
rgb_out[:, :, 1] = green_interp_final
rgb_out = np.clip(rgb_out, 0.0, 1.0) 


# plt.imshow(rgb_out)
# plt.axis('off')  # Hide axis ticks
# plt.show()

########Color correction
camera_matrix = np.array([[6988,-1384,-714],[-5631,13410,2447],[-1485,2204,7318]])/10000
srgb_matrix = np.array([[0.4124564, 0.3575761, 0.1804375], 
                        [0.2126729, 0.7151522, 0.0721750], 
                        [0.0193339, 0.1191920, 0.9503041]])

color_correction = np.linalg.inv(camera_matrix @ srgb_matrix)

color_correction = color_correction / color_correction.sum(axis=1, keepdims=True)

image = np.tensordot(rgb_out, color_correction, ([2],[1]))
image = np.clip(image, 0.0, 1.0)


############Manual White Balancing
plt.imshow(image)
plt.axis('off')  # Hide axis ticks
points = plt.ginput(n=2, timeout=30) 
y_min = min(int(points[0][0]), int(points[1][0]))
y_max = max(int(points[0][0]), int(points[1][0]))
x_min = min(int(points[0][1]), int(points[1][1]))
x_max = max(int(points[0][1]), int(points[1][1]))
print(points)
print(x_min, x_max, y_min, y_max)
print(image.shape)
print((image[x_min:x_max, y_min:y_max, :]).shape)
r_avg = np.average(image[x_min:x_max, y_min:y_max, 0])
g_avg = np.average(image[x_min:x_max, y_min:y_max, 1])
b_avg = np.average(image[x_min:x_max, y_min:y_max, 2])
image[:,:,0] *= g_avg/r_avg
image[:,:,2] *= g_avg/b_avg
print(r_avg, g_avg, b_avg)
image = np.clip(image, 0.0, 1.0)


#############Brightening
image = image*2
mean_gs = 0.12
image = image*mean_gs/np.average(skimage.color.rgb2gray(image))
image = np.clip(image, 0, 1)
#print(np.average(skimage.color.rgb2gray(image)))

###############Gamma corrrection 
image = np.piecewise(image, [image <= 0.0031308, image > 0.0031308], [lambda x: 12.92*x, lambda x: 1.055*x**(1/2.4)-0.055])


##########Save Image
skimage.io.imsave("campus_stair_world.png", skimage.img_as_ubyte(image))

# plt.imshow(image)
# plt.axis('off')  # Hide axis ticks
# plt.show()