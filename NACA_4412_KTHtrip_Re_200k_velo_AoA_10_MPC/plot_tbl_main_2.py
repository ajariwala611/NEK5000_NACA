from paraview.simple import *
import argparse
import os
from mpi4py import MPI

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def view1(cam):
    cam.CameraPosition = [0.7, 0.6, 0.8]
    cam.CameraFocalPoint = [0.7,0.0,-0.1]
    # cam.CameraViewUp = [-3.082312207058858e-05, 0.409252687295691, -0.9124210853499555]
    # cam.CameraViewAngle = 1.202713723883428
    # cam.CameraParallelScale = 0.50
    # cam.Yaw(0)
    # cam.Roll(0)
    # cam.Zoom(1.0)

def view2(cam, dz=1.0):
    cam.SetPosition(15,35,1.26)     # Camera position
    cam.SetFocalPoint(15,0,1.25)    # Camera focal point
    cam.Zoom(dz)

def view3(cam, dz=1.0):
    """ Zoom in closer in time """
    cam.SetPosition(-55,20,25)     # Camera position
    cam.SetFocalPoint(7,0.3,1.2)    # Camera focal point
    cam.Zoom(dz)

def view4(cam, r=1.0):
    """ Move away in time """
    focal_point = [7,0.3,1.2]
    end = [-55.,20.,25.]
    rmin = 0.25 # Start from half-way toward the focal point
    # r = 1: camera position == end; r = 0: camera position == focal_point
    r = rmin + (1. - rmin) * r / 600.
    cam.SetPosition(
        end[0] * r + focal_point[0] * (1. - r),
        end[1] * r + focal_point[1] * (1. - r),
        end[2] * r + focal_point[2] * (1. - r),
    )     # Camera position
    cam.SetFocalPoint(*focal_point)    # Camera focal point

def view5(cam, frame, num_frames):
    """ Move linearly in time from initial position to end position """
    initial_pos = [0., 2., 1.25]
    focal_point_initial = [30, -8, 1.25]
    end_pos = [20., 2., 1.25]
    focal_point_end = [50, -8, 1.25]

    r = frame / num_frames  # Calculate interpolation parameter

    # Interpolate camera position
    cam.SetPosition(
        initial_pos[0] * (1 - r) + end_pos[0] * r,
        initial_pos[1] * (1 - r) + end_pos[1] * r,
        initial_pos[2] * (1 - r) + end_pos[2] * r
    )

    # Interpolate focal point
    cam.SetFocalPoint(
        focal_point_initial[0] * (1 - r) + focal_point_end[0] * r,
        focal_point_initial[1] * (1 - r) + focal_point_end[1] * r,
        focal_point_initial[2] * (1 - r) + focal_point_end[2] * r
    )

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = 'data_every_0.002_500_files/NACA.nek5000'
output = 'frames/frame'

# view = GetActiveView()
#   

# create a new 'Nek5000 Reader'
# nACAnek5000 =  OpenDataFile(data)
# nACAnek5000 =  VisItNek5000Reader(registrationName='NACA.nek5000', FileName=data)
nACAnek5000 =  Nek5000Reader(registrationName='NACA.nek5000', FileName=data)
# nACAnek5000.PointArrays = ['x_velocity', 'temperature', 's2','pressure']
nACAnek5000.PointArrays = ['Velocity','Velocity Magnitude']
# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# Properties modified on nACAnek5000
nACAnek5000.MergePointstoCleangrid = 1

# calculator1 = Calculator(registrationName='Calculator1', Input=nACAnek5000)
# # Properties modified on calculator1
# calculator1.ResultArrayName = 'Ux'
# calculator1.Function = 'Velocity_X'

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=nACAnek5000)
# Properties modified on contour1
contour1.ContourBy = ['POINTS', 'Velocity Magnitude']
contour1.Isosurfaces = [1e-06]
# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, False)
# set scalar coloring
ColorBy(contour1Display, None)

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=nACAnek5000)

# Properties modified on resampleToImage1
resampleToImage1.UseInputBounds = 0
resampleToImage1.SamplingDimensions = [2000, 200, 100]
resampleToImage1.SamplingBounds = [0.05, 1.2, 0.0, 0.2, 0.0, 0.05]
resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')
resampleToImage1Display.SetRepresentationType('Volume')
ColorBy(resampleToImage1Display, ('POINTS', 'Velocity', 'X'))
# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction('Velocity')
velocityLUT.ApplyPreset('Cold and Hot', True)
velocityLUT.RescaleTransferFunction(-0.45, 1.9)
velocityLUT.RGBPoints = [-0.45, 0.0, 1.0, 1.0, 0.5941979289054871, 0.0, 0.0, 1.0, 0.7247227072715761, 0.0, 0.0, 0.501960784314, 0.8421949779987337, 1.0, 0.0, 0.0, 1.9, 1.0, 1.0, 0.0]
velocityPWF = GetOpacityTransferFunction('Velocity')
velocityPWF.Points = [-0.45, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.3543146252632141, 0.0, 0.5, 0.0, 0.85, 0.5, 0.5, 0.0, 0.85, 0.0, 0.5, 0.0, 1.9, 0.0, 0.5, 0.0]
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Orientation = 'Horizontal'
velocityLUTColorBar.Title = "$u_x$"
velocityLUTColorBar.ComponentTitle = ''

# # create a new 'Contour'
# contour2 = Contour(registrationName='Contour2', Input=nACAnek5000)
# # Properties modified on contour2
# contour2.ContourBy = ['POINTS', 'Temperature']
# contour2.Isosurfaces = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')
# # trace defaults for the display properties.
# contour2Display.Representation = 'Surface'
# # get color transfer function/color map for 'temperature'
# temperatureLUT = GetColorTransferFunction('Temperature')
# # get opacity transfer function/opacity map for 'temperature'
# temperaturePWF = GetOpacityTransferFunction('Temperature')
# # get 2D transfer function for 'temperature'
# temperatureTF2D = GetTransferFunction2D('Temperature')
# # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
# # temperatureLUT.ApplyPreset('X Ray', True)
# # invert the transfer function
# # temperatureLUT.InvertTransferFunction()
# contour2Display.AmbientColor = [0.0, 0.0, 0.0]
# contour2Display.DiffuseColor = [0.0, 0.0, 0.0]
# contour2Display.Opacity = 0.2

# # create a new 'Contour' Upwash for v'
# contour3 = Contour(registrationName='Contour3', Input=nACAnek5000)
# # Properties modified on contour3
# contour3.ContourBy = ['POINTS', 'S02']
# contour3.Isosurfaces = [0.12]
# # show data in view
# contour3Display = Show(contour3, renderView1, 'GeometryRepresentation')
# # trace defaults for the display properties.
# contour3Display.Representation = 'Surface'
# # show color bar/color legend
# contour3Display.SetScalarBarVisibility(renderView1, False)
# ColorBy(contour3Display, ('POINTS', 'S02'))

# # create a new 'Contour' downwash for v'
# contour4 = Contour(registrationName='Contour4', Input=nACAnek5000)
# # Properties modified on contour3
# contour4.ContourBy = ['POINTS', 'S02']
# contour4.Isosurfaces = [-0.15]
# # show data in view
# contour4Display = Show(contour4, renderView1, 'GeometryRepresentation')
# # trace defaults for the display properties.
# contour4Display.Representation = 'Surface'
# # show color bar/color legend
# contour4Display.SetScalarBarVisibility(renderView1, True)
# ColorBy(contour3Display, ('POINTS', 'S02'))

# s02LUT = GetColorTransferFunction('S02')
# s02LUT.RescaleTransferFunction(-0.2, 0.2)
# s02LUTColorBar = GetScalarBar(s02LUT, renderView1)
# s02LUTColorBar.Orientation = 'Horizontal'
# s02LUTColorBar.Title = "$v'$"
# s02PWF = GetOpacityTransferFunction('S02')
# s02PWF.RescaleTransferFunction(-0.2, 0.2)
# s02TF2D = GetTransferFunction2D('S02')
# s02TF2D.RescaleTransferFunction(-0.2, 0.2, 0.0, 1.0)


# create a new 'Gradient'
# gradient1 = Gradient(registrationName='Gradient1', Input=nACAnek5000)
# gradient1.ScalarArray = ['POINTS', 'Velocity']
# gradient1.ComputeQCriterion = 1
# contour1 = Contour(registrationName='Contour1', Input=gradient1)
# contour1.ContourBy = ['POINTS', 'Q Criterion']
# contour1.Isosurfaces = [1000.0]
# contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
# contour1Display.Representation = 'Surface'
# ColorBy(contour1Display, ('POINTS', 'Velocity', 'X'))
# # get color transfer function/color map for 'Velocity'
# velocityLUT = GetColorTransferFunction('Velocity')
# velocityLUT.RGBPoints = [-0.2, 0.231373, 0.298039, 0.752941, 0.0, 0.865003, 0.865003, 0.865003, 1.5, 0.705882, 0.0156863, 0.14902]
# velocityPWF = GetOpacityTransferFunction('Velocity')
# velocityTF2D = GetTransferFunction2D('Velocity')
# velocityLUT.RGBPoints = [-0.2, 0.231373, 0.298039, 0.752941, 0.0, 1.0, 0.865003, 0.865003, 1.5, 0.705882, 0.0156863, 0.14902]
# velocityLUT.RGBPoints = [-0.2, 0.231373, 0.298039, 0.752941, 0.0, 1.0, 1.0, 0.865003, 1.5, 0.705882, 0.0156863, 0.14902]
# velocityLUT.RGBPoints = [-0.2, 0.231373, 0.298039, 0.752941, 0.0, 1.0, 1.0, 1.0, 1.5, 0.705882, 0.0156863, 0.14902]
# velocityLUT.RescaleTransferFunction(-0.2, 1.5)
# velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
# velocityLUTColorBar.Orientation = 'Horizontal'
# velocityLUTColorBar.Title = "$u_x$"
# velocityLUTColorBar.ComponentTitle = ''



animate = 1  #to loop through all the times
parallel = 1 # To loop in parallel 
    
timestep = 1 # To only plot one frame, set animate and prallel = 0 for this 

if animate:
    total_num_steps = len(nACAnek5000.TimestepValues)
    if parallel:
        # Calculate the number of steps per rank
        steps_per_rank = total_num_steps // size
        remainder = total_num_steps % size

        # Determine the range of timestep values for the current rank
        start_step = rank * steps_per_rank + min(rank, remainder)
        end_step = (rank + 1) * steps_per_rank + min(rank + 1, remainder)

        # Generate the timestep values for the current rank
        timestep_values = range(start_step, end_step)
        print(f'Rank {rank+1} of {size} will process {len(list(timestep_values))} frames: {timestep_values}')
    else:
        timestep_values = range(total_num_steps)
        print(f'Rank {rank+1} of {size} will process {len(list(timestep_values))} frames: {timestep_values}')
else:
    timestep_values = [timestep]


for timestep in timestep_values:
    print('Plotting time step ', timestep)
    # renderView1.ViewTime = reader.TimestepValues[timestep]
    view = GetActiveView()
    view.ViewTime = nACAnek5000.TimestepValues[timestep]
    # Render again and save
    view1(view)
    Render()

    # SaveScreenshot(f'{output}_time_{timestep:06d}.png', renderView1, ImageResolution=[4*1920, 4*1080],OverrideColorPalette='BlackBackground')
    SaveScreenshot(f'{output}_time_{timestep:06d}.png', renderView1, ImageResolution=[4*1920, 4*1080],CompressionLevel='5', FontScaling='Scale fonts proportionally')
    # SaveScreenshot(f'{args.output}_time_{timestep:06d}.png', view, ImageResolution=[4*1920, 4*1080],OverrideColorPalette='BlackBackground',CompressionLevel='3')