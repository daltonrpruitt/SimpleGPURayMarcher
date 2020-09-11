'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Initial Setup for Ray Marcher
'''

import moderngl
import moderngl_window
import numpy as np
from window_setup import BasicWindow

#Experimenting with inputs
#import input_commands 





class RayMarchingWindow(BasicWindow):
    gl_version = (3, 3)
    title = "Ray Marching Demo Scene"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
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

        self.prog['sphere.center'].value = (2.0, 0.0, 8.0)
        self.prog['sphere.radius'].value = 1.0
        self.prog['sphere.color'].value = (1.0, 0.0, 0.0)
        self.prog['sphere.shininess'].value = 32.0

        self.prog['box_center'].value = (-1, 0, 8)
        self.prog['box_rotation'].value = (0, 0, 0) # Degrees

        self.prog['back_color'].value = (0, 0.3, 0.9, 1.0)

        self.prog['light'].value = (2., 2., 5., 1.)
        self.prog['light_color'].value = (1., 1., 1.)

        self.prog['cam_pos'].value = (0, 0, -5)

        self.prog['using_point_light'] = True
        self.prog['using_dir_light'].value = True

        vertices = np.array([
            -1, -1,
            1, -1,
            -1, 1,
            -1, 1,
            1, -1,
            1, 1

        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def render(self, time, frame_time):
        bc = self.prog['back_color'].value
        self.ctx.clear(bc[0], bc[1], bc[2], bc[3],)
        # Reset these uniforms here to resize the scene before rendering again
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        self.vao.render()
        self.prog['sphere.center'].value = ( np.cos(time/2)*2, np.sin(time/2)*3, 8.0 + np.sin(time/2) *2)  #np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
        self.prog['box_center'].value = ( np.cos(time/2-np.pi)*2, 0, 8.0 + np.sin(time/2-np.pi ) * 2)  #np.cos(time), np.sin(time*1.5), 10.0 + np.sin(time/2) * 5, 0.5
        self.prog['box_rotation'].value = (0, -time, 0)  #
        #self.prog['cam_pos'].value = (0, 1+ np.sin(time)*0.7, -5)


if __name__ == '__main__':
    RayMarchingWindow.run()

