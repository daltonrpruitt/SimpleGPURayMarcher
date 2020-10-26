'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Initial Setup for Ray Marcher
'''

#import moderngl
from moderngl_window import screenshot
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
            fragment_shader=open("a4.frag", "r").read(),

        )
        
        self.animate = False
        self.scene_number = 0
        self.scene_time = 0

        self.show_sphere = True
        self.show_box = True
        self.show_crate = False
        self.show_link = True
        self.using_point_light = True
        self.using_direction_light = False
        self.using_sphere_light = False

        self.antialiasing_sample_frequency = 1
        self.use_depth_of_field = False

        self.prog['u_gloss_blur_coeff'].value = 0.3
        self.prog['u_gloss_samples'].value = 5

        self.prog['tonemap_reinhard'].value = False
        self.prog['tonemap_exposure'].value = False
        self.prog['exposure'].value = 1


        self.lens_parameters = ['u_focal_distance', 'u_lens_distance','u_lens_radius']
        self.current_lens_paramter = self.lens_parameters[0]
        self.prog['u_focal_distance'].value = 9
        #self.prog['u_lens_distance'].value = 1
        self.prog['u_lens_radius'].value = 0.20
        self.prog['u_dof_samples'].value = 4
        
        
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        # self.prog['time'].value = 0

        self.prog['antialiasing_sample_frequency'].value = self.antialiasing_sample_frequency
        self.prog['max_recursive_depth'].value = 5
        self.prog['use_depth_of_field'].value = self.use_depth_of_field

        self.prog['sphere.center'].value = (1.7, 0, 3.0)
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





        self.prog['box_center'].value = (-1, 0, 1)
        self.prog['box_rotation'].value = (0, np.pi/4, 0) # Degrees
        
        # Crate SDF
        crate_res = 512
        self.prog['crate_scale'].value = 1
        self.prog['crate_center'].value = (1.5, 0.5, 5*self.prog['crate_scale'].value)
        self.prog['crate_rotation'].value = (np.pi/4, np.pi/2, np.pi/6)

        # Toon Link SDF
        link_res = 512
        self.link_scale = 1

        self.offscreen_pos = (0, -100, -100)

        self.hide_objects()

        self.prog['back_color'].value = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)

        self.prog['point_light.position'].value = (1., 2, 1.) 
        self.prog['point_light.color'].value = (1., 1., 1.)
        self.prog['point_light.intensity'].value = 1.
        #self.prog['point_light.direction'].value = (0,0,0)

        direction = (1., -1., -1.)
        dist = np.sqrt(sum(x**2 for x in direction))
        normalized_dir = tuple([d / dist for d in direction])
        self.prog['directional_light.direction'].value = normalized_dir
        self.prog['directional_light.color'].value = (1., 1., 1.)
        self.prog['directional_light.intensity'].value = 1.
        #self.prog['diretional_light.position'].value = (0,0,0)

        self.prog['volLight.center'].value = (2, 3., 0.)
        self.prog['volLight.radius'].value = 2.
        self.prog['volLight.color'].value = (1.0, 1.0, 1.0)
        self.prog['volLight.intensity'].value = 1.0

        self.prog['cam_pos'].value = (0, 0, -5)

        self.prog['using_point_light'].value = self.using_point_light
        self.prog['using_dir_light'].value = self.using_direction_light
        self.prog['using_sphere_light'].value = self.using_sphere_light

        # Some debugging print statements
        '''for name in self.prog:
            member = self.prog[name]
            if name.find("link") > -1:
                print(name, member.value)'''
            

        sphere_pos = self.prog['sphere.center'].value
        cam_pos = self.prog['cam_pos'].value
        line_to_sphere = (sphere_pos[0]- cam_pos[0], sphere_pos[1]- cam_pos[1], sphere_pos[2] - cam_pos[2])
        link_to_sphere_ratio = 10
        link_pos = tuple(line_to_sphere[i]*link_to_sphere_ratio + cam_pos[i] for i in range(len(line_to_sphere)))
        #self.prog['linkSDFInfo'].value = (1, (0, 1, 8), (0.,0.,0.), 1.0, (0.1, 0.5, 0.7),  4.0 )
        self.prog['linkSDFInfo.id'].value = 1
        self.prog['linkSDFInfo.position'].value = link_pos # (-1.5, 0.1, (1+5)*self.link_scale - 5)
        self.prog['linkSDFInfo.rotation'].value = (0, -np.pi/2, 0.)
        self.prog['linkSDFInfo.scale'].value =  self.link_scale
        self.prog['linkSDFInfo.color'] =  (0.2, 0.2, 0.7)
        self.prog['linkSDFInfo.shininess'] =  16.0
        self.prog['linkSDFInfo.reflectiveness'] =  0.9
        self.prog['linkSDFInfo.is_transparent'] =  True
        #linksdf.reflectiveness=  0.0
        '''for name in self.prog:
            member = self.prog[name]
            if name.find("link") > -1:
                print(name, member.value)'''

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

        self.screenshot_buffer = self.ctx.screen
        self.has_screenshot = False
        self.take_screenshot = False

        self.keep_rendering = True


    def hide_objects(self):
        if not self.show_sphere:
            self.prog['sphere.center'].value = self.offscreen_pos

        if not self.show_box:
            self.prog['box_center'].value = self.offscreen_pos

        if not self.show_crate:
            self.prog['crate_center'].value = self.offscreen_pos

        if not self.show_link:
            self.prog['linkSDFInfo.position'].value = self.offscreen_pos




    def set_scene(self, scene_num):
        '''
        I got tired of changing numbers everytime I wanted to see how something 
        looked in a certain scene setup. Here are the setups to choose from:
        0 : Sphere and box floating around  
        '''
        if scene_num == 0:
            self.scene_number = 0

            self.show_sphere = True
            self.show_box = True
            self.show_crate = False
            self.show_link = False 
            self.hide_objects()
            
            self.prog['sphere.center'].value = (2, 0 , 8)
            self.prog['box_center'].value = (-2, 0 , 8)

        if scene_num == 1:
            self.scene_number = 1

            self.show_sphere = False
            self.show_box = False
            self.show_crate = True
            self.show_link = True 
            self.hide_objects()
            self.prog['linkSDFInfo.position'].value = (2, 0 , 8)
            self.prog['crate_center'].value = (-2, 0.5 , 8)
       
        if scene_num == 2:
            self.scene_number = 2

            self.show_sphere = True
            self.show_box = False
            self.show_crate = False    
            self.show_link = True 
            self.hide_objects()
            self.scene_time = 0
            self.prog['linkSDFInfo.position'].value = (0, 0 , -20)
            self.prog['sphere.center'].value = (0, 0 , 8)

            
        


    def animate_objects(self, i_time, frame_time):
        '''
        Moves stuff around according to which scene is active
        '''
        if self.scene_number == 0 :
            self.prog['sphere.center'].value =(np.cos(i_time/2)*2, 0, 8.0 + np.sin(i_time/2) *2)  
            self.prog['box_center'].value = ( np.cos(i_time/2-np.pi)*2, 0, 8.0 + np.sin(i_time/2 - np.pi) * 2)  
            self.prog['box_rotation'].value = (i_time/3, -1*i_time/2, 0)  

        if self.scene_number == 1:
            self.prog['linkSDFInfo.rotation'].value =  ( (np.pi/8, i_time, np.pi/8) ) 
            self.prog['crate_rotation'].value =  ( (np.pi/8, i_time, np.pi/8) ) 

        if self.scene_number == 2:
            self.scene_time += frame_time
            self.prog['linkSDFInfo.position'].value =  (0, 0, -20 + (self.scene_time)**2 ) 



    def key_event(self, key, action, modifiers):
        
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:

            if key == self.wnd.keys.Q:
                self.keep_rendering = not self.keep_rendering


            if key == self.wnd.keys.A:
                self.animate = not self.animate

            if key == self.wnd.keys.F1:
                self.set_scene(0)            
                
            if key == self.wnd.keys.F2:
                self.set_scene(1)
            
            if key == self.wnd.keys.F3:
                self.set_scene(2)


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
                step = 0.1
                if self.current_lens_paramter == self.lens_parameters[0]:
                    step = 1
                self.prog[self.current_lens_paramter].value += step
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )
            if key == self.wnd.keys.MINUS:
                step = 0.1
                if self.current_lens_paramter == self.lens_parameters[0]:
                    step = 1
                self.prog[self.current_lens_paramter].value -= step
                print(self.current_lens_paramter,"=",self.prog[self.current_lens_paramter].value )

            if key == self.wnd.keys.SPACE:
                self.prog['use_depth_of_field'].value = not self.prog['use_depth_of_field'].value
           
            if key == self.wnd.keys.T:
                self.prog['sphere.is_transparent'].value = not self.prog['sphere.is_transparent'].value
           

            if key == self.wnd.keys.G and not modifiers.shift and not modifiers.ctrl:
                if self.prog['sphere.glossiness'].value < 1 :
                    self.prog['sphere.glossiness'].value += 0.1
                print("Sphere glossiness:", self.prog['sphere.glossiness'].value)

            if key == self.wnd.keys.G and modifiers.shift and not modifiers.ctrl:
                if self.prog['sphere.glossiness'].value > 0 :
                    self.prog['sphere.glossiness'].value -= 0.1
                print("Sphere glossiness:", self.prog['sphere.glossiness'].value)
            

            if key == self.wnd.keys.G and modifiers.ctrl and not modifiers.shift:
                if self.prog['plane.glossiness'].value <  1 :
                    self.prog['plane.glossiness'].value += 0.1
                print("Plane glossiness:", self.prog['plane.glossiness'].value)

            if key == self.wnd.keys.G and modifiers.ctrl and modifiers.shift:
                if self.prog['plane.glossiness'].value > 0 :
                    self.prog['plane.glossiness'].value -= 0.1
                print("Plane glossiness:", self.prog['plane.glossiness'].value)

            if key == self.wnd.keys.L:
                if not modifiers.ctrl:
                    if not modifiers.shift:
                        self.prog['point_light.intensity'].value *= 1.25
                    else: 
                        self.prog['point_light.intensity'].value /= 1.25
                    print("Point Light Intensity =", round(self.prog['point_light.intensity'].value,2))
                if modifiers.ctrl:
                    if not modifiers.shift:
                        self.prog['volLight.intensity'].value *= 1.25
                    else: 
                        self.prog['volLight.intensity'].value /= 1.25
                    print("Volume Light Intensity =", round(self.prog['volLight.intensity'].value,2))


            # Tone Mapping Commands
            if key == self.wnd.keys.M:
                if not modifiers.ctrl:
                    if not modifiers.shift:
                        self.prog['tonemap_reinhard'].value = not self.prog['tonemap_reinhard'].value 
                    else:
                        self.prog['tonemap_exposure'].value = not self.prog['tonemap_exposure'].value 
                else:
                    if self.prog['tonemap_reinhard'].value ^ self.prog['tonemap_exposure'].value:
                        self.prog['tonemap_reinhard'].value = not self.prog['tonemap_reinhard'].value
                        self.prog['tonemap_exposure'].value = not self.prog['tonemap_exposure'].value
                print("Tonemapping : Reinghard =", self.prog['tonemap_reinhard'].value, "Exposure =", self.prog['tonemap_exposure'].value )



            if key == self.wnd.keys.PERIOD:
                self.prog['exposure'].value *= 1.5
                print("Exposure =", round(self.prog['exposure'].value,2))
            if key == self.wnd.keys.COMMA:
                self.prog['exposure'].value /= 1.5
                print("Exposure =", round(self.prog['exposure'].value,2))

            if key == self.wnd.keys.UP:
                old_val = self.prog['sphere.center'].value
                self.prog['sphere.center'].value = (old_val[0], old_val[1],old_val[2]+1)

            if key == self.wnd.keys.DOWN:
                old_val = self.prog['sphere.center'].value
                self.prog['sphere.center'].value = (old_val[0], old_val[1],old_val[2]-1)   

            if key == self.wnd.keys.S and not modifiers.ctrl:
                print("Switching soft shadows to", not self.prog['using_sphere_light'].value)
                self.prog['using_sphere_light'].value = not self.prog['using_sphere_light'].value
                self.prog['using_point_light'].value = not self.prog['using_point_light'].value

            if key == self.wnd.keys.S and modifiers.ctrl:
                print("Screenshot Buffer is", type(self.screenshot_buffer))
                #print("moderngl_window.screenshot is", type(screenshot))
                screenshot.create(self.screenshot_buffer, file_format='png')


            # Using modifiers (shift and ctrl)

            if key == self.wnd.keys.Z and modifiers.shift:
                print("Shift + Z was pressed")

            if key == self.wnd.keys.Z and modifiers.ctrl:
                print("ctrl + Z was pressed")

        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            pass
        
    def render(self, run_time, frame_time):
        if not self.keep_rendering:
            return

        bc = self.prog['back_color'].value
        self.ctx.clear(bc[0], bc[1], bc[2], bc[3],)
        # Reset these uniforms here to resize the scene before rendering again
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height

        #self.prog['crate_rotation'].value = (np.pi/4, time, np.pi/4)
        
        if not self.has_screenshot and self.take_screenshot:
            print("Taking Screenshot")
            start = time.time()

        self.vao.render()

        if not self.has_screenshot and self.take_screenshot:
            self.has_screenshot = True
            screenshot.create(self.screenshot_buffer, file_format='png')           
            print("Screenshot took " +"{:.3}".format(time.time() - start)+"s to create")

        #self.prog['crate_center'].value = (0, 0, 7+np.cos(time)*3 )
        #self.prog['light'].value = (1, 3+np.cos(time/2)*4, 0 , 1)
        
        if self.animate:
            self.animate_objects(run_time, frame_time)
 

if __name__ == '__main__':
    RayMarchingWindow.run()
    #wind = RayMarchingWindow()
    #wind.render()
