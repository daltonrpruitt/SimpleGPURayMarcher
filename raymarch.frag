#version 330
uniform float width;
uniform float height;
// uniform float time;
uniform vec4 sphere;
uniform vec4 sphere_color;
uniform vec4 back_color;
uniform vec4 light;
uniform vec3 light_color;

out vec4 f_color;

float rand(vec2);
vec4 shade(vec4);

void main() {
    vec3 cam_pos = vec3(0.0, 0.0, -5.0);
    vec2 window_size = vec2(width,height);
    //f_color = ray;
    
    float signed_dist = 1.0e6;
    f_color = back_color;
    vec4 color = vec4(0,0,0,0);
    float ambient_coeff = 0.4;
    
    // From page 310 of Shirley and Marschner
    int sample_frequency = 2; // 3x3
    for(int p = 0; p < sample_frequency; p++) {  
        for(int q = 0; q < sample_frequency; q++) {  
        
            // Make rays offset uniform amounts from center
            //vec2 sub_region = vec2( (j % 2) * 2.0 - 1.0, (j / 2) * 2.0 - 1.0 ) * 0.5;
            //vec2 sub_pixel = gl_FragCoord.xy + sub_region + vec2(random_shift)/2;
            
            vec2 sub_region = vec2((p+0.5)/sample_frequency - 0.5, (q+0.5)/sample_frequency - 0.5);
            float random_shift = rand(vec2(gl_FragCoord.xy + sub_region));
            vec2 sub_pixel = gl_FragCoord.xy + sub_region + vec2(random_shift)/sample_frequency;
            
            vec4 ray = vec4(normalize(vec3((sub_pixel - window_size/2.0)/height, -cam_pos.z)), 1.0);
            for(int i = 0; i < 32; i++) {
                // TODO: Loop over objects
                signed_dist = length(ray.xyz * ray.w + cam_pos - sphere.xyz ) - (sphere.w ) ;
                if(signed_dist < 0.0001 ){
                
                    vec3 amb_color = ambient_coeff * sphere_color.rgb;
                    
                    // TODO: Make shade() function, fix current math...
                    vec3 pos_on_sphere = cam_pos + ray.xyz * ray.w;
                    vec3 norm = normalize(pos_on_sphere - sphere.xyz);
                    vec3 vec_to_light = normalize(light.xyz - pos_on_sphere);
                    float lambertian = clamp(dot(vec_to_light, norm), 0.0, 1.0);
                    
                    vec3 diffuse_color = light_color.rgb * lambertian * sphere_color.rgb;
                    
                    f_color = vec4(diffuse_color, 1.0);
                    
                    // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
                    // Check if is actually reflecting the correct way
                    vec3 reflected_vec = reflect(-vec_to_light, norm);
                    //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
                    vec3 e_vec = normalize(-1.0 * ray.xyz);  // negative so facing correct way
                    float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
                    vec3 specular_color = light.w * light_color * pow(e_dot_r, 16.0);

                    color = color + vec4( amb_color + diffuse_color + specular_color, 1.0);
                    //f_color = vec4( diffuse_color + specular_color, 1.0);
                    
                    break;
                } else if ( i >= 31 ){
                    //f_color = back_color;
                    // break;
                    color = color + back_color;
                } else {
                    ray.w = ray.w + signed_dist;
                }
            }
        }
    }
    f_color = clamp(color / pow(sample_frequency, 2.0), vec4(0.0), vec4(1.0));;
    

}

// from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}
