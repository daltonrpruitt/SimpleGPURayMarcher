
'''
    Renders a plaid triangle (animated)
'''

import numpy as np

from window_setup import BasicWindow

class HelloWorld(BasicWindow):
    gl_version = (3, 3)
    title = "Hello World"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                
                
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    
                }
            ''',
            fragment_shader='''
                #version 330
                uniform float width;
                uniform float height;
                uniform float time;
                out vec4 f_color;
                void main() {
                    f_color = vec4((sin(time*(gl_FragCoord.x + width/2.0)/width)+1.0)/2.0 , 
                                    (sin(0.5*time*(gl_FragCoord.y + height/2.0)/height)+1.0)/2.0, 0.8, 1.0);
                }
            ''',
        )
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        self.prog['time'].value = 0;


        vertices = np.array([
            0.0, 0.8,
            -0.6, -0.8,
            0.6, -0.8,


        ], dtype='f4')
        ''' -0.8, 0.8,
                    0.8, -0.8,
                    0.8, 0.8,'''
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def render(self, time, frame_time):
        self.ctx.clear(0.2, 0.4, 0.95)
        self.vao.render()
        self.prog['time'].value += 0.5


if __name__ == '__main__':
    HelloWorld.run()

