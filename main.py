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
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        # self.prog['time'].value = 0

        self.prog['sphere.center'].value = (2.0, 2.0, 8.0)
        self.prog['sphere.radius'].value = 0.0
        self.prog['sphere.color'].value = (1.0, 0.0, 0.0)
        self.prog['sphere.shininess'].value = 32.0

        self.prog['box_center'].value = (-1, 0, 8)
        self.prog['box_rotation'].value = (0, 0, 0) # Degrees
        
        # Crate SDF
        self.prog['crate_center'].value = (0, 0, 8)
        self.prog['crate_rotation'].value = (0, 5, 0)
        self.prog['crate_scale'].value = 2 

        self.prog['back_color'].value = (0, 0.3, 0.9, 1.0)

        self.prog['light'].value = (2., 3, 5., 1.)
        self.prog['light_color'].value = (1., 1., 1.)

        self.prog['cam_pos'].value = (0, 2, -5)

        self.prog['using_point_light'] = True
        self.prog['using_dir_light'].value = False


        self.animate = False
        
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
        crate_res = 64
        try:
            file_name = "SDFs/crate_voxels_res-"+str(crate_res)
            crate_voxel_sdf = np.loadtxt(file_name,delimiter=",",dtype="float32")
        except OSError as e:
            print("No crate file with resolution=" +str(crate_res),"Actual error:",e)
            print("Exiting...")       
            exit()
        #print(crate_voxel_sdf.shape)
        # Reshape to 3D
        crate_voxel_sdf = np.reshape(crate_voxel_sdf, newshape=(crate_res+2, crate_res + 2, crate_res + 2))
        #print(crate_voxel_sdf.shape)
        
        # Remove padding on each of the dimensions (padded with 1's); also have to order to C_continguous for the texture creation below
        crate_voxel_sdf = crate_voxel_sdf.copy(order='C')#crate_voxel_sdf[1:-1, 1:-1, 1:-1].copy(order='C')

        self.crate_sdf_texture = self.ctx.texture3d(size=(crate_res+2, crate_res+2, crate_res+2), components=1, data=crate_voxel_sdf,dtype="f4")
        self.crate_sdf_buffer = self.ctx.buffer(reserve=np.size(crate_voxel_sdf))
        self.crate_sdf_texture.read_into(self.crate_sdf_buffer)
        #print(self.crate_sdf_buffer.size)

        self.vbo = self.ctx.buffer(vertices)

        #self.prog['crate_sdf_buffer'].value = self.crate_sdf_buffer

        self.vao = self.ctx.vertex_array(
            self.prog, 
            [
                (self.vbo, '2f', 'in_vert'),
                #(self.crate_sdf_buffer, str(crate_res)+"f", 'crate_sdf_texture'),
            ],
            idx_buffer
        )


    def render(self, time, frame_time):
        bc = self.prog['back_color'].value
        self.ctx.clear(bc[0], bc[1], bc[2], bc[3],)
        # Reset these uniforms here to resize the scene before rendering again
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        self.vao.render()
        self.crate_sdf_texture.use()
        self.prog['crate_rotation'].value = (0, time/3, 0)
        if self.animate:
            self.prog['sphere.center'].value = ( np.cos(time/2)*2, np.sin(time/2)*3, 8.0 + np.sin(time/2) *2)  #np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
            self.prog['box_center'].value = ( np.cos(time/2-np.pi)*2, 0, 8.0 + np.sin(time/2-np.pi ) * 2)  #np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
            self.prog['box_rotation'].value = (0, -time, 0)  #
            #self.prog['cam_pos'].value = (0, 1+ np.sin(time)*0.7, -5)


if __name__ == '__main__':
    RayMarchingWindow.run()

