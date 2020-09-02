import moderngl
import moderngl_window

'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Initial Setup for Ray Marcher
'''

import numpy as np

from window_setup import BasicWindow

class RayMarchingWindow(BasicWindow):
    gl_version = (3, 3)
    title = "Hello World"

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
            fragment_shader=open("raymarch.frag","r").read(),

        )
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        #self.prog['time'].value = 0
        self.prog['sphere'].value = (0.0, 0.0, 15.0, 0.5)
        self.prog['sphere_color'].value = (1.0, 0.0, 0.0, 1.0)
        self.prog['back_color'].value = (1, 1, 1, 1.0)
        self.prog['light'].value = (2, 2, 10, 1)
        self.prog['light_color'].value = (1, 1, 1)


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
        self.prog['sphere'].value = (np.cos(time), np.sin(time*1.5), 15.0 + np.sin(time/1.5) * 7, 0.5) #


if __name__ == '__main__':
    RayMarchingWindow.run()

