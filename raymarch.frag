#version 330

struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
};


struct Plane {
    vec3 normal;
    float distance;
    vec3 color;
};
 
//Plane information
vec3 plane_norm = normalize(vec3(0., 1, -0.0));
float plane_dist = 0.5;
vec3 plane_color = vec3(0.2431, 0.7451, 0.0431);

Plane plane = Plane(plane_norm, plane_dist, plane_color);

uniform float width;
uniform float height;
// uniform float time;
uniform Sphere sphere;
//uniform vec4 sphere_color;
uniform vec4 back_color;
uniform vec4 light;
uniform vec3 light_color;
uniform bool using_dir_light;


out vec4 f_color;



int march_iterations = 1024;
vec3 cam_pos = vec3(0.0, 0.0, -5.0);
float ambient_coeff = 0.4;
float maxDistance = 1.0e8;
vec3 dir_light = vec3(1, -1, 1);
vec3 dir_light_color = vec3(0.5, 0.2, 0.9);

float rand(vec2);
vec3 shade(vec4, vec3, vec3, vec3 );
float sdfSphere(Sphere, vec4);
float sdfPlane(Plane, vec4 );
void marchRay(out int, inout vec4);

void main() {

    vec2 window_size = vec2(width,height);
    //f_color = ray;
    
    f_color = back_color;
    vec4 color = vec4(0,0,0,0);
    
    // From page 310 of Shirley and Marschner
    int sample_frequency = 2; // 
    //for(int p = 0; p < sample_frequency; p++) {  
        //for(int q = 0; q < sample_frequency; q++) {  
        
            // Make rays offset uniform amounts from center
            
            //vec2 sub_region = vec2((p+0.5)/sample_frequency - 0.5, (q+0.5)/sample_frequency - 0.5);
            //float random_shift = rand(vec2(gl_FragCoord.xy + sub_region));
            //vec2 sub_pixel = gl_FragCoord.xy + sub_region + vec2(random_shift)/sample_frequency;
            
            vec4 ray = vec4(normalize(vec3((gl_FragCoord.xy - window_size/2.0)/height*abs(cam_pos.z),-cam_pos.z)-cam_pos), 1.0);
            //for(int i = 0; i < march_iterations; i++) {
                // TODO: Loop over objects
                int object_hit = -1;
                marchRay(object_hit, ray);
                
                /*
                float sphere_signed_dist = length(ray.xyz * ray.w + cam_pos - sphere.center ) - (sphere.radius ) ;
                
                float plane_signed_dist = sdfPlane(plane, ray);

                bool sphere_closer;
                if(sphere_signed_dist <= plane_signed_dist){
                    signed_dist = sphere_signed_dist;
                    sphere_closer = true;
                } else {
                    signed_dist = plane_signed_dist;
                    sphere_closer = false;
                    //f_color = vec4((-dot(ray.xyz , plane_norm)-0.1)*2.0);return;   //Debug
                }*/
                if(object_hit == -1) {
                    // Hit nothing
                    f_color = back_color; 
                    return;
                }else if (object_hit == 0) {
                    // Sphere
                    vec3 p_hit = cam_pos + ray.xyz * ray.w;
                    f_color = vec4(shade(ray, p_hit, normalize(p_hit - sphere.center), sphere.color), 1.0);
                    return;
                } else if(object_hit == 1){
                    vec3 p_hit = cam_pos + ray.xyz * ray.w;
                    vec3 shaded_color = shade(ray, p_hit, plane_norm, plane_color);
                    f_color = vec4(shaded_color, 1.0);
                    return;
                } else {
                    // Error
                    f_color = vec4(0.);
                }
                /*
                if(signed_dist < 0.0001 ){
                
                    if(sphere_closer) {
                        vec3 p_hit = cam_pos + ray.xyz * ray.w;
                        f_color = vec4(shade(ray, p_hit, normalize(p_hit - sphere.center), sphere.color), 1.0);
                        return;
                       

                    } else {
                        vec3 p_hit = cam_pos + ray.xyz * ray.w;
                        vec3 shaded_color = shade(ray, p_hit, plane_norm, plane_color);
                        f_color = vec4(shaded_color, 1.0);
                        return;

                    }
        
                } else {
                    ray.w = ray.w + signed_dist;
                }
            }
            f_color = back_color;*/
       // }
    //}

    //f_color = clamp(color / pow(sample_frequency, 2.0), vec4(0.0), vec4(1.0));;
    

}



void marchRay(out int object_hit, inout vec4 ray){
    
    object_hit = -1;
    for(int i = 0; i < march_iterations; i++) {
        // TODO: Loop over objects
        float signed_dist = maxDistance;

        float sphere_signed_dist = sdfSphere(sphere, ray);
        
        float plane_signed_dist = sdfPlane(plane, ray);

        int object_closer = -1;
        //bool sphere_closer;
        if(sphere_signed_dist <= plane_signed_dist){
            signed_dist = sphere_signed_dist;
            object_closer = 0;
        } else {
            signed_dist = plane_signed_dist;
            object_closer = 1;
            //f_color = vec4((-dot(ray.xyz , plane_norm)-0.1)*2.0);return;   //Debug
        }

        if(signed_dist < 0.0001 ){
            object_hit = object_closer;
            return;
        } else if (signed_dist >= maxDistance) {
            object_hit = -1; 
            return;
        } else { 
            ray.w = ray.w + signed_dist;
        }
    }
    return;
    
}

float sdfPlane(Plane plane, vec4 ray){
    if(dot(ray.xyz , plane.normal) < 0){
        // plane facing camera
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        vec3 p_plane = plane.normal * -plane.distance;
        return dot(cam_pos + ray.xyz * ray.w - p_plane, plane.normal) + plane.distance; 
    } else {
        return maxDistance;
    }
}
float sdfSphere(Sphere sph, vec4 ray){
    return length(ray.xyz * ray.w + cam_pos - sph.center) - sph.radius;
}

// Based on general structure of Dr. TJ Jankun-Kelly's Observable Notes: https://observablehq.com/@infowantstobeseen/basic-ray-marching
vec3 shade(vec4 original_ray, vec3 hit_point, vec3 normal, vec3 object_color) {
    vec3 amb_color = ambient_coeff * object_color;
    
    if(!using_dir_light){

        // TODO: Make shade() function, fix current math...
        vec3 vec_to_light = normalize(light.xyz - hit_point);
        float lambertian = clamp(dot(vec_to_light, normal), 0.0, 1.0);
        
        vec3 diffuse_color = light_color * lambertian * object_color;
                                
        // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
        // Check if is actually reflecting the correct way
        vec3 reflected_vec = reflect(-vec_to_light, normal);
        //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
        vec3 e_vec = normalize(-1.0 * original_ray.xyz);  // negative so facing correct way
        float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
        vec3 specular_color = light.w * light_color * pow(e_dot_r, 4.0);
    
        return vec3(amb_color + diffuse_color + specular_color);
    } else {
        // TODO: Make shade() function, fix current math...
        vec3 vec_to_light = -dir_light;
        float lambertian = clamp(dot(vec_to_light, normal), 0.0, 1.0);
        
        vec3 diffuse_color = dir_light_color * lambertian * object_color;
                                
        // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
        // Check if is actually reflecting the correct way
        vec3 reflected_vec = reflect(-vec_to_light, normal);
        //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
        vec3 e_vec = normalize(-1.0 * original_ray.xyz);  // negative so facing correct way
        float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
        vec3 specular_color = light.w * dir_light_color * pow(e_dot_r, 4.0);
    
        return vec3(amb_color + diffuse_color + specular_color);
    }
}


// from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}
