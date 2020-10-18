'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Initial Setup for Ray Marcher
'''

#import moderngl
#import moderngl_window
import numpy as np
from window_setup import BasicWindow  # imports moderngl_window

#Experimenting with inputs
#import input_commands 
from moderngl import Uniform
import time



class RayMarchingWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Ray Marching Demo Scene"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 430
                in vec2 in_vert;
                //uniform float width, height;
                //uniform vec4 sphere;
                
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    
                }
            ''',
            fragment_shader=open("raymarch.frag", "r").read(),

        )
        
        self.animate = False

        self.show_sphere = True
        self.show_box = True
        self.show_crate = False
        self.show_link = True
        self.using_point_light = True
        self.using_direction_light = False
        self.using_sphere_light = False

        self.antialiasing_sample_frequency = 1
        self.use_depth_of_field = True

        self.prog['u_gloss_blur_coeff'].value = 0.3

        self.lens_parameters = ['u_focal_distance', 'u_lens_distance','u_lens_radius']
        self.current_lens_paramter = self.lens_parameters[0]
        self.prog['u_focal_distance'].value = 13
        #self.prog['u_lens_distance'].value = 1
        self.prog['u_lens_radius'].value = 0.07
        self.prog['u_dof_samples'].value = 20

        
        
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        # self.prog['time'].value = 0

        self.prog['antialiasing_sample_frequency'].value = self.antialiasing_sample_frequency
        self.prog['max_recursive_depth'].value = 5
        self.prog['use_depth_of_field'].value = self.use_depth_of_field

        self.prog['sphere.center'].value = (0.0, 0, 7.0)
        self.prog['sphere.radius'].value = 1
        self.prog['sphere.color'].value = (1.0, 0.0, 0.0)
        self.prog['sphere.shininess'].value = 32.0
        self.prog['sphere.reflectiveness'].value = 0.4
        self.prog['sphere.is_transparent'].value = False
        self.prog['sphere.glossiness'].value = 0.0

        plane_norm = np.array((0, 1,0))
        plane_norm = plane_norm / np.sqrt(plane_norm[0]**2+plane_norm[1]**2+plane_norm[2]**2)
        print(plane_norm)
        self.prog['plane.normal'].value = tuple(plane_norm)
        self.prog['plane.distance'].value = 1
        self.prog['plane.color'].value = (0.2431, 0.7451, 0.0431)
        self.prog['plane.shininess'].value = 16.0
        self.prog['plane.reflectiveness'].value = 0.5
        #self.prog['plane.is_transparent'].value = False
        self.prog['plane.glossiness'].value = 00.0





        self.prog['box_center'].value = (-2, 0, 2.5)
        self.prog['box_rotation'].value = (0, np.pi/4, 0) # Degrees
        
        # Crate SDF
        crate_res = 512
        self.prog['crate_scale'].value = 1
        self.prog['crate_center'].value = (1.5, 0.5, 5*self.prog['crate_scale'].value)
        self.prog['crate_rotation'].value = (np.pi/4, np.pi/2, np.pi/6)

        # Toon Link SDF
        link_res = 512
        self.link_scale = 1

        offscreen_pos = (0, -100, -100)



        self.prog['back_color'].value = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)

        self.prog['light'].value = (0., 2, 0., 1.)
        self.prog['light_color'].value = (1., 1., 1.)


        self.prog['volLight.center'].value = (1.5, 5., 0.)
        self.prog['volLight.radius'].value = 2.
        self.prog['volLight.color'].value = (1.0, 1.0, 1.0)
        #self.prog['volLight.intensity'].value = 1.0

        self.prog['cam_pos'].value = (0, 0, -5)

        self.prog['using_point_light'].value = self.using_point_light
        self.prog['using_dir_light'].value = self.using_direction_light
        self.prog['using_sphere_light'].value = self.using_sphere_light

        # Some debugging print statements
        '''for name in self.prog:
            member = self.prog[name]
            if name.find("link") > -1:
                print(name, member.value)'''
            
        #self.prog['linkSDFInfo'].value = (1, (0, 1, 8), (0.,0.,0.), 1.0, (0.1, 0.5, 0.7),  4.0 )
        self.prog['linkSDFInfo.id'].value = 1
        self.prog['linkSDFInfo.position'].value = (0.0, 0.0, 20.0) #(-1.5, 0.1, (1+5)*self.link_scale - 5)
        self.prog['linkSDFInfo.rotation'].value = (0, -np.pi/2, 0.)
        self.prog['linkSDFInfo.scale'].value =  self.link_scale
        self.prog['linkSDFInfo.color'] =  (0.1, 0.5, 0.7)
        self.prog['linkSDFInfo.shininess'] =  16.0
        self.prog['linkSDFInfo.reflectiveness'] =  0.9
        #linksdf.reflectiveness=  0.0
        '''for name in self.prog:
            member = self.prog[name]
            if name.find("link") > -1:
                print(name, member.value)'''

        if not self.show_sphere:
            self.prog['sphere.center'].value = offscreen_pos

        if not self.show_box:
            self.prog['box_center'].value = offscreen_pos

        if not self.show_crate:
            self.prog['crate_center'].value = offscreen_pos

        if not self.show_link:
            self.prog['linkSDFInfo.position'].value = offscreen_pos



        vertices = np.array([
            -1, -1,
            1, -1,
            -1, 1,
            1, 1

        ], dtype='f4')

        # https://github.com/moderngl/moderngl/blob/master/examples/raymarching.py
        idx_data = np.array([
            0, 1, 2,
            2, 1, 3
        ])
        idx_buffer = self.ctx.buffer(idx_data)


        # Construct VBO for SDF Voxel Sampler

        print("Beginning loading crate...",end="\t")
        start = time.time()
        try:
            try:
                file_name = "SDFs/crate_voxels_res-"+str(crate_res)
                crate_voxel_sdf = np.loadtxt(file_name,delimiter=",",dtype="float32")
            except OSError:
                file_name = "SDFs/crate_voxels_nopad_depthSign_res-"+str(crate_res)
                crate_voxel_sdf = np.loadtxt(file_name,delimiter=",",dtype="float32")
        except OSError as e:
            try:
                try:
                    file_name = "SDFs/crate_voxels_res-"+str(crate_res)+".npy"
                    crate_voxel_sdf = np.load(file_name).astype(np.float32)
                except OSError:
                    file_name = "SDFs/crate_voxels_nopad_depthSign_res-"+str(crate_res)+".npy"
                    crate_voxel_sdf = np.load(file_name).astype(np.float32)            
            except OSError: 
                print("No Crate file with resolution=" +str(crate_res),"Actual error:",e)
                print("Exiting...")       
                exit()
        
        #self.crate_sdf_texture = self.ctx.texture3d(size=(crate_res+2, crate_res+2, crate_res+2), components=1, data=crate_voxel_sdf,dtype="f4")

        # Reshape to 3D, create texture, create buffer, read texture data into buffer
        crate_voxel_sdf = np.reshape(crate_voxel_sdf, newshape=(crate_res, crate_res , crate_res ), order='C')
        #crate_voxel_sdf = crate_voxel_sdf[1:-1,1:-1,1:-1].copy( )# get rid of padding
        self.crate_sdf_texture = self.ctx.texture3d(size=(crate_res, crate_res, crate_res), components=1, data=crate_voxel_sdf,dtype="f4")
        self.crate_sdf_buffer = self.ctx.buffer(reserve=np.size(crate_voxel_sdf))
        self.crate_sdf_texture.read_into(self.crate_sdf_buffer)
        #print(self.crate_sdf_buffer.size)
        print(f"crate={time.time() - start:.2f}s",end="\t")

        start = time.time()
        try:
            try:
                file_name = "SDFs/ToonLink_voxels_res-"+str(link_res)
                link_voxel_sdf = np.loadtxt(file_name,delimiter=",",dtype="float32")
            except OSError:
                file_name = "SDFs/ToonLink_voxels_nopad_depthSign_res-"+str(link_res)
                link_voxel_sdf = np.loadtxt(file_name,delimiter=",",dtype="float32")
        except OSError as e:
            try:
                try:
                    file_name = "SDFs/ToonLink_voxels_res-"+str(link_res)+".npy"
                    link_voxel_sdf = np.load(file_name, allow_pickle=True)
                except OSError:
                    file_name = "SDFs/ToonLink_voxels_nopad_depthSign_res-"+str(link_res)+".npy"
                    link_voxel_sdf = np.load(file_name, allow_pickle=True)
            except OSError as e:
                print("No Toon Link file with resolution=" +str(link_res),"Actual error:",e)
                print("Exiting...")       
                exit()
        # Reshape to 3D, create texture, create buffer, read texture data into buffer
        # No Padding
        try:
            link_voxel_sdf = np.reshape(link_voxel_sdf, newshape=(link_res, link_res, link_res), order='C') # No Padding
            print("reshaped link sdf without padding")
            try:
                self.link_sdf_texture = self.ctx.texture3d(size=(link_res, link_res, link_res), components=1, data=link_voxel_sdf,dtype="f4")
            except:
                self.link_sdf_texture = self.ctx.texture3d(size=(link_res, link_res, link_res), components=1, data=link_voxel_sdf.astype(np.float32),dtype="f4")
        except:
            print()
            try:
                link_voxel_sdf = np.reshape(link_voxel_sdf, newshape=(link_res+2, link_res+2, link_res+2 ), order='C') 
                self.link_sdf_texture = self.ctx.texture3d(size=(link_res+2, link_res+2, link_res+2), components=1, data=link_voxel_sdf,dtype="f4")
            except:
                print("Error setting up link sdf into texture. Exiting...")
                exit()
        self.link_sdf_buffer = self.ctx.buffer(reserve=np.size(link_voxel_sdf))
        self.link_sdf_texture.read_into(self.link_sdf_buffer)

        print(f"Link={time.time() - start:.2f}s")


        self.crate_sdf_texture.use(location=0) # first
        self.link_sdf_texture.use(location=1) # second

        self.vbo = self.ctx.buffer(vertices)

        self.vao = self.ctx.vertex_array(
            self.prog, 
            [
                (self.vbo, '2f', 'in_vert'),
                #(self.crate_sdf_buffer, str(crate_res)+"f", 'crate_sdf_texture'),
            ],
            idx_buffer
        )

    def scene_operation(self, scene_num, operation="set"):
        '''
        I got tired of changing numbers everytime I wanted to see how something 
        looked in a certain scene setup. Here are the setups to choose from:
        0 : Sphere and box floating around  
        '''
        if scene_num == 0:
            if operation == "set":
                self.show_sphere = True
                self.show_box = True
                self.show_crate = False
                self.show_link = False 

                #self.prog['sphere.center'] = 0.
            



    def key_event(self, key, action, modifiers):
    
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.NUMBER_1:
                self.current_lens_paramter = self.lens_parameters[0]
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )

            if key == self.wnd.keys.NUMBER_2:
                self.current_lens_paramter = self.lens_parameters[1]
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )

            if key == self.wnd.keys.NUMBER_3:
                self.current_lens_paramter = self.lens_parameters[2]
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )

            if key == self.wnd.keys.EQUAL:
                self.prog[self.current_lens_paramter].value += 0.1
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )
            if key == self.wnd.keys.MINUS:
                self.prog[self.current_lens_paramter].value -= 0.1
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )

            if key == self.wnd.keys.SPACE:
                self.prog['use_depth_of_field'].value = not self.prog['use_depth_of_field'].value
           
            if key == self.wnd.keys.T:
                self.prog['sphere.is_transparent'].value = not self.prog['sphere.is_transparent'].value

            # Using modifiers (shift and ctrl)

            if key == self.wnd.keys.Z and modifiers.shift:
                print("Shift + Z was pressed")

            if key == self.wnd.keys.Z and modifiers.ctrl:
                print("ctrl + Z was pressed")

        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            pass
        
    def render(self, time, frame_time):
        bc = self.prog['back_color'].value
        self.ctx.clear(bc[0], bc[1], bc[2], bc[3],)
        # Reset these uniforms here to resize the scene before rendering again
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height

        #self.prog['crate_rotation'].value = (np.pi/4, time, np.pi/4)
        
        self.vao.render()

        #self.prog['crate_center'].value = (0, 0, 7+np.cos(time)*3 )
        #self.prog['light'].value = (1, 3+np.cos(time/2)*4, 0 , 1)
        
        if self.animate:
            if self.show_sphere:
                self.prog['sphere.center'].value =(np.cos(time/2)*2, 0, 8.0 + np.sin(time/2) *2)  # (1, 0, 8.0 + np.sin(time/2) *2)    np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
            if self.show_box:
                self.prog['box_center'].value = ( np.cos(time/2-np.pi*2/3)*2, 0, 8.0 + np.sin(time/2 - np.pi*2/3) * 2)  #np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
                self.prog['box_rotation'].value = (time/3, -time/2, 0)  #
            if self.show_link:
                self.prog['linkSDFInfo.position'] =  ( np.cos(time/2+np.pi*2/3)*2, 0, 8.0 + np.sin(time/2 + np.pi*2/3) * 2)
                self.prog['linkSDFInfo.rotation'] =  ( (np.pi/8, time, np.pi/8) ) 

            #self.prog['cam_pos'].value = (0, 1+ np.sin(time)*0.7, -5)


if __name__ == '__main__':
    RayMarchingWindow.run()
    #wind = RayMarchingWindow()
    #wind.render()
