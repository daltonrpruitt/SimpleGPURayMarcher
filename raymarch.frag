#version 330

struct Sphere {

    vec3 center;
    vec3 color;
    float radius;

};

//Plane information
vec3 plane_norm = normalize(vec3(0., 0., -1.));
float plane_dist = -20.0;
vec3 plane_color = vec3(0.5);

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
    //for(int p = 0; p < sample_frequency; p++) {  
        //for(int q = 0; q < sample_frequency; q++) {  
        
            // Make rays offset uniform amounts from center
            
            //vec2 sub_region = vec2((p+0.5)/sample_frequency - 0.5, (q+0.5)/sample_frequency - 0.5);
            //float random_shift = rand(vec2(gl_FragCoord.xy + sub_region));
            //vec2 sub_pixel = gl_FragCoord.xy + sub_region + vec2(random_shift)/sample_frequency;
            
            vec4 ray = vec4(normalize(vec3((gl_FragCoord.xy - window_size/2.0)/height, -cam_pos.z)), 1.0);
            for(int i = 0; i < 32; i++) {
                // TODO: Loop over objects
                float sphere_signed_dist = length(ray.xyz * ray.w + cam_pos - sphere.xyz ) - (sphere.w ) ;
                
                float plane_signed_dist = 1.0e6;
                if(dot(ray.xyz, plane_norm) < 0){
                    plane_signed_dist = dot(cam_pos + ray.xyz * ray.w, plane_norm) - plane_dist; //(plane_dist - dot(cam_pos, plane_norm))/dot(ray.xyz, plane_norm);
                }

                bool sphere_closer;
                if(sphere_signed_dist < plane_signed_dist){
                    signed_dist = sphere_signed_dist;
                    sphere_closer = true;
                } else {
                    signed_dist = plane_signed_dist;
                    sphere_closer = false;

                }

                if(signed_dist < 0.0001 ){
                
                    if(sphere_closer) {
                    vec3 amb_color = ambient_coeff * sphere_color.rgb;
                    
                    // TODO: Make shade() function, fix current math...
                    vec3 hit_pos = cam_pos + ray.xyz * ray.w;
                    vec3 hit_norm = normalize(hit_pos - sphere.xyz);
                    vec3 vec_to_light = normalize(light.xyz - hit_pos);
                    float lambertian = clamp(dot(vec_to_light, hit_norm), 0.0, 1.0);
                    
                    vec3 diffuse_color = light_color.rgb * lambertian * sphere_color.rgb;
                    
                    //f_color = vec4(diffuse_color, 1.0);
                    
                    // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
                    // Check if is actually reflecting the correct way
                    vec3 reflected_vec = reflect(-vec_to_light, hit_norm);
                    //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
                    vec3 e_vec = normalize(-1.0 * ray.xyz);  // negative so facing correct way
                    float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
                    vec3 specular_color = light.w * light_color * pow(e_dot_r, 16.0);

                    //color = color + vec4( amb_color + diffuse_color + specular_color, 1.0);
                    f_color = vec4( diffuse_color + specular_color, 1.0);

                    } else {
                        // Plane hit
                        vec3 amb_color = ambient_coeff * plane_color;
                    
                        // TODO: Make shade() function, fix current math...
                        vec3 hit_pos = cam_pos + ray.xyz * ray.w;
                        vec3 hit_norm = plane_norm;
                        vec3 vec_to_light = normalize(light.xyz - hit_pos);
                        float lambertian = clamp(dot(vec_to_light, hit_norm), 0.0, 1.0);
                        
                        vec3 diffuse_color = light_color.rgb * lambertian * plane_color;
                                                
                        // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
                        // Check if is actually reflecting the correct way
                        vec3 reflected_vec = reflect(-vec_to_light, hit_norm);
                        //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
                        vec3 e_vec = normalize(-1.0 * ray.xyz);  // negative so facing correct way
                        float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
                        vec3 specular_color = light.w * light_color * pow(e_dot_r, 16.0);

                        //color = color + vec4( amb_color + diffuse_color + specular_color, 1.0);
                        f_color = vec4( diffuse_color + specular_color, 1.0);

                    }
                    break;
        
                } else if ( i >= 31 ){
                    //f_color = back_color;
                    // break;
                    color = color + back_color;
                } else {
                    ray.w = ray.w + signed_dist;
                }
            }
       // }
    //}

    //f_color = clamp(color / pow(sample_frequency, 2.0), vec4(0.0), vec4(1.0));;
    

}

// Based on general structure of Dr. TJ Jankun-Kelly's Observable Notes: https://observablehq.com/@infowantstobeseen/basic-ray-marching
vec3 shade(/*some object struct*/vec4 object, vec3 point, vec3 to_eye) {
    


    
    return vec3(-1., -1., -1.);
}


// from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}
