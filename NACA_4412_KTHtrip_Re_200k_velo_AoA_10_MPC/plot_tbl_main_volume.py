from paraview.simple import *
import argparse
import os
from mpi4py import MPI

def view1(cam):
    # cam.SetPosition(0.55, 0.1, 1.1)    # Camera position
    # cam.SetFocalPoint(0.55, 0.1, 0.025)    # Camera focal point    # cam.Elevation(0)
    cam.SetPosition(0.5000000000000007, 1.6477535753607895, 0.9110123811410831)
    cam.SetFocalPoint(0.5000000000000007, 0.03910909664226343, -0.01773894167746351)
    cam.SetViewUp(0.0, 0.5000000000000002, -0.8660254037844386)
    cam.SetViewAngle(17.759914255091104)
    # cam.Yaw(0)
    # cam.Roll(0)
    cam.Zoom(1.0)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paraview animation script for the Nek5000 turbulent boundary layer DNS.')
    parser.add_argument('--data', default='flat_plate_reg.nek5000')
    parser.add_argument('--output', default='flat_plate')
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=0)
    parser.add_argument('--animate', action='store_true', default=False)
    parser.add_argument('--q-criterion', action='store_true', default=False)
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--slice', action='store_true', default=False)
    parser.add_argument('--iso-u', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--volume', action='store_true', default=False)
  
    args = parser.parse_args()

    # Load data file
#   reader =  OpenDataFile('/Volumes/ResearchDat/nek5000/Nek5000/run/turbulent_flow_control/turbulent_boundary_layer/statistics/flat_plate.nek5000')

    if not os.path.exists(args.data):
        print('Could not locate data file')

    reader = OpenDataFile(args.data)

    reader.PointArrays = ['x_velocity', 's2']
    # reader.MergePointstoCleangrid = 1
    reader.UpdatePipeline()

    annotateTime = AnnotateTimeFilter(reader)
    Show(annotateTime)

    Render() # First render to make zoom work at the second render

    view = GetActiveView()
    cam = GetActiveCamera()

    if args.volume:
        resampleToImage1 = ResampleToImage(Input=reader)
        resampleToImage1.UseInputBounds = 0
        resampleToImage1.SamplingDimensions = [1500, 50, 125]
        resampleToImage1.SamplingBounds = [0.0, 30.0, 0.0, 1.0, 0.0, 2.5]
        # print('Interpolated data onto image grid')
        SetActiveSource(resampleToImage1)

        dp = Show(resampleToImage1)
        dp.SetRepresentationType('Volume')
        ColorBy(dp,('POINTS','x_velocity'))
        LUT = GetColorTransferFunction('x_velocity')
        LUT.ApplyPreset('Black, Blue and White', True)
        PWF = GetOpacityTransferFunction('x_velocity')

        LUT.RescaleTransferFunction(0.0, 0.6)
        PWF.RescaleTransferFunction(0.0, 0.6)

        PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.14509804546833038, 1.0, 0.5, 0.0, 0.51, 0.8644859790802002, 0.5, 0.0, 0.6, 0.0, 0.5, 0.0]
        LUT.RGBPoints = [0.0, 0.0, 0.0, 0.0, 0.41, 0.0, 0.0, 0.11764705882352941, 0.44509807229042053, 0.0, 0.501960784314, 1.0, 0.6, 1.0, 1.0, 1.0]
    if args.iso_u:
        """ u iso surface near the wall """

        u = 0.11    # Change u isosurface value here

        # Contours
        iso_u = Contour(reader)
        iso_u.ContourBy = 's2'
        iso_u.Isosurfaces = [u]

        # Contour Coloring
        color_by = 'x_velocity'
        color_range = [0.0, 1.3]

        dp = GetDisplayProperties(iso_u)
        dp.Representation = 'Surface'
        dp.LookupTable = GetColorTransferFunction(color_by)
        dp.LookupTable.ApplyPreset('Cool to Warm (Extended)', True)
        dp.LookupTable.RescaleTransferFunction(color_range)
        dp.ColorArrayName = color_by

        Show(iso_u) 

        # Contours
        foil = Contour(reader)
        foil.ContourBy = 'x_velocity'
        foil.Isosurfaces = [1e-6]

        dp_foil = Show(foil) 
        ColorBy(dp_foil, None)

    if args.slice:

        slice = Slice(reader)
        slice.SliceType.Normal = [0,0,1]
        slice.SliceType.Origin = [5,0,0.025]

        # Clip Coloring
        color_by = 'x_velocity'
        color_range = [-0.01, 1.3]

        dp = GetDisplayProperties(slice)
        dp.Representation = 'Surface'
        dp.LookupTable = GetColorTransferFunction(color_by)
        dp.LookupTable.ApplyPreset('Rainbow Uniform', True)
        dp.LookupTable.RescaleTransferFunction(color_range)
        dp.ColorArrayName = color_by

        Show(slice)

    if args.clip:
        """ Near-wall clip """

        height = 0.05    # Change height here

        clip = Clip(reader)
        clip.ClipType.Normal = [0,1,0]
        clip.ClipType.Origin = [0,height,0]

        # Clip Coloring
        color_by = 'x_velocity'
        color_range = [0.1, 1.0]

        dp = GetDisplayProperties(clip)
        dp.Representation = 'Surface'
        dp.LookupTable = GetColorTransferFunction(color_by)
        dp.LookupTable.ApplyPreset('Cool to Warm', True)
        dp.LookupTable.RescaleTransferFunction(color_range)
        dp.ColorArrayName = color_by

        Show(clip)

    if args.q_criterion:
        """ Q-criterion """

        # Create Q Grid
        #q_grid_initial = FastUniformGrid()
        #q_grid_initial.WholeExtent = [0, 3000, 0, 100, 0, 250]
        #q_grid_initial.GenerateSwirlVectors = 0
        #q_grid_initial.GenerateDistanceSquaredScalars = 0
        #q_grid_initial.UpdatePipeline()
        #q_grid = Transform(q_grid_initial)
        #q_grid.Transform.Scale = [0.01, 0.01, 0.01]
        #q_grid.UpdatePipeline()

        #print('Greated Q-criterion grid')

        # Resample to q grid
        #q_grid_data = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=q_grid)
        # resampleToImage1 = ResampleToImage(Input=reader)
        # resampleToImage1.UseInputBounds = 0
        # resampleToImage1.SamplingDimensions = [3000, 100, 250]
        # resampleToImage1.SamplingBounds = [0.0, 30.0, 0.0, 1.0, 0.0, 2.5]
        # print('Interpolated velocity to Q-criterion Image grid')
        # ----- Q Criterion -----
        gradient = Gradient(resampleToImage1)
        gradient.ScalarArray = 'velocity'
        gradient.ComputeGradient = 0
        gradient.ComputeQCriterion = 1
        gradient.QCriterionArrayName = 'q_criterion'

        # Contours
        contour = Contour(gradient)
        contour.ContourBy = 'q_criterion'
        contour.Isosurfaces = [5.0]

        # Contour Coloring
        color_by = 'velocity'
        color_range = [0.1, 1.0]
        dp = Show(gradient)
        dp.SetRepresentationType('Outline')
        dp = GetDisplayProperties(contour)
        dp.Representation = 'Surface'
        dp.LookupTable = GetColorTransferFunction(color_by)
        dp.LookupTable.ApplyPreset('Cool to Warm (Extended)', True)
        dp.LookupTable.RescaleTransferFunction(color_range)
        dp.ColorArrayName = color_by

        Show(contour)
    # Select times to animate - Split accross mpi tasks
    # if args.animate:
    #     total_num_steps = len(reader.TimestepValues)
    #     if args.parallel:
    #         dt = total_num_steps // size + 1
    # #       timestep_values = range(rank*dt,(rank+1)*dt) if rank < size-1 else range(rank*dt,total_num_steps)
    #         timestep_values = range(min(rank*dt,total_num_steps),min((rank+1)*dt, total_num_steps))
    #         print(f'rank {rank+1} of {size} will plot frames ', timestep_values)
    #     else:
    #         timestep_values = range(total_num_steps)
    # else:
    #     timestep_values = [args.timestep]
    # Select times to animate - Split accross mpi tasks
    if args.animate:
        total_num_steps = len(reader.TimestepValues)
        if args.parallel:
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
    else:
        timestep_values = [args.timestep]

#   timestep_values = range(len(reader.TimestepValues)) if args.animate else [args.timestep]

    # Set camera zoom before iterations
    if args.view == 2:
        min_zoom = 8.
        max_zoom = 1.
        dz = (max_zoom / min_zoom) ** (1. / (total_num_steps-1))
        if len(timestep_values) > 0:
            cam.Zoom(min_zoom * dz**timestep_values[0])
    elif args.view == 3:
        min_zoom = 2.
        max_zoom = 8.
        dz = (max_zoom / min_zoom) ** (1. / (total_num_steps-1))
        if len(timestep_values) > 0:
            cam.Zoom(min_zoom * dz**timestep_values[0])
    elif args.view == 4:
        cam.Zoom(3.)

    for timestep in timestep_values:
        print('Plotting time step ', timestep)
        view.ViewTime = reader.TimestepValues[timestep]
        view.ViewSize = [1920, 1080]

        # Camera Position
        if args.view == 1:
            view1(cam)
        elif args.view == 2:
            view2(cam, dz=dz)
        elif args.view == 3:
            view3(cam, dz=dz)
        elif args.view == 4:
            view4(cam,total_num_steps, r=timestep)
        elif args.view == 5:
            view5(cam,timestep,total_num_steps)
        else:
            print('View ', args.view, ' is not available. Choosing defalut view.')
            cam = GetActiveCamera()

        # Render again and save
        Render()

        SaveScreenshot(f'{args.output}_time_{timestep:06d}.png', view, ImageResolution=[4*1920, 4*1080],OverrideColorPalette='BlackBackground')
        # SaveScreenshot(f'{args.output}_time_{timestep:06d}.png', view, ImageResolution=[4*1920, 4*1080],OverrideColorPalette='BlackBackground',CompressionLevel='3')


