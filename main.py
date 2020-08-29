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
            fragment_shader='''
                #version 330
                uniform float width;
                uniform float height;
                // uniform float time;
                uniform vec4 sphere;
                uniform vec4 sphere_color;
                uniform vec4 back_color;
                uniform vec4 light;
                
                out vec4 f_color;
                
                vec4 shade(vec4);
                void main() {
                    vec3 cam_pos = vec3(0.0, 0.0, -5.0);
                    vec2 window_size = vec2(width,height);
                    //f_color = ray;
                    
                    float signed_dist = 1.0e6;
                    f_color = back_color;
                    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
                    
                    for(int j = 0; j < 4; j++) {
                        // Make 4 rays offset uniform amount from center
                       // vec2 sub_pixel = gl_FragCoord.xy + vec2( (j % 2) * 2.0 - 1.0, (j // 2) * 2.0 - 1.0 ) * 0.5;
                        vec4 ray = vec4(normalize(vec3((gl_FragCoord.xy - window_size/2.0)/height, -cam_pos.z)), 1.0);
                        for(int i = 0; i < 32; i++) {
                            // Loop over objects
                            signed_dist = length(ray.xyz * ray.w + cam_pos - sphere.xyz ) - (sphere.w ) ;
                            if(signed_dist < 0.001 ){
                            
                                
                                // TODO: Make shade() function, fix current math...
                                vec3 pos_on_sphere = cam_pos + ray.xyz * ray.w;
                                vec3 norm = normalize(pos_on_sphere - sphere.xyz);
                                vec3 vec_to_light = normalize(light.xyz - pos_on_sphere);
                                float lambertian = dot(vec_to_light, norm);
                                vec3 diffuse_color = max(lambertian,0.0) * sphere_color.rgb;
                                   
                                // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
                                // Check if is actually reflecting the correct way
                                vec3 reflected_vec = 2.0 * dot(vec_to_light, norm) * norm - vec_to_light;
                                vec3 e_vec = -1.0 * ray.xyz;  // negative so facing correct way
                                float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
                                vec3 specular_color = light.w * light.rgb * pow(e_dot_r, 8.0);
    
                                color = color + clamp(vec4(diffuse_color + specular_color, 0.0), vec4(0.0), vec4(1.0));
                                
                            } else {
                                ray.w = ray.w + signed_dist;
                            }
                        }
                    }
                    f_color = color / 4;
                    

                }
                
                
            ''',
        )
        self.prog['width'].value = self.wnd.width
        self.prog['height'].value = self.wnd.height
        #self.prog['time'].value = 0
        self.prog['sphere'].value = (0.0, 0.0, 15.0, 1.0)
        self.prog['sphere_color'].value = (0.5, 0.2, 0.7, 1.0)
        self.prog['back_color'].value = (0.2, 0.4, 0.95, 1.0)
        self.prog['light'].value = (3, 3, 10, 1)


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
        self.prog['sphere'].value = (np.cos(time)/2.0, np.sin(time)/2.0, 15.0 + np.sin(time*2)*3, 1.0)


if __name__ == '__main__':
    RayMarchingWindow.run()

