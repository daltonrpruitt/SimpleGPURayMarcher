#version 330

struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float shininess;
};


struct Plane {
    vec3 normal;
    float distance;
    vec3 color;
    float shininess;
};
 
//Plane information
vec3 plane_norm = normalize(vec3(0., 1, -0));
float plane_dist = 1;
vec3 plane_color = vec3(0.2431, 0.7451, 0.0431);
float plane_shininess = 16.0;

Plane plane = Plane(plane_norm, plane_dist, plane_color, plane_shininess);

uniform float width;
uniform float height;
// uniform float time;
uniform Sphere sphere;
uniform vec4 back_color;
uniform vec4 light;
uniform vec3 light_color;
uniform bool using_dir_light;


out vec4 f_color;



int march_iterations = 1024;
vec3 cam_pos = vec3(0.0, 0.0, -3.0);
float ambient_coeff = 0.4;
float maxDistance = 1.0e3;
vec3 dir_light = vec3(1, -1, 1);
vec3 dir_light_color = vec3(0.5, 0.2, 0.9);

float rand(vec2);
vec3 shade(vec4, vec3, vec3, vec3, float);
float sdfSphere(Sphere, vec4, vec3);
float sdfPlane(Plane, vec4, vec3);
void marchRay(out int, inout vec4, in vec3, float);
vec3 getNormal(vec3, int);
//void marchShadow(out int, in vec4 );

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
            
            vec4 ray = vec4(normalize(vec3((gl_FragCoord.xy - window_size/2.0)/height*abs(cam_pos.z),-cam_pos.z)-cam_pos), 0.001);
            //for(int i = 0; i < march_iterations; i++) {
            // TODO: Loop over objects
            int object_hit;
            marchRay(object_hit, ray, cam_pos, maxDistance);
            
            if(object_hit == -1) {
                // Hit nothing
                f_color = back_color; 
                return;
            } 

            vec3 p_hit = cam_pos + ray.xyz*ray.w;
            vec3 obj_normal = getNormal(p_hit, object_hit);
            //f_color = vec4(obj_normal*vec3(2, 2, 3), 1); 
            vec3 to_light;
            float max_dist = maxDistance; 
            if(!using_dir_light){
                to_light = normalize(light.xyz - p_hit);
                max_dist = length(light.xyz - p_hit);
            } else {
                to_light = -1.0 * dir_light;
            }
            vec4 shadow_ray = vec4(to_light, 0.001);
            //f_color = vec4(shadow_ray.r*2, shadow_ray.g * 2, shadow_ray.b*2, 1.0); 
            //return;
            int obj_in_way;
            marchRay(obj_in_way, shadow_ray, p_hit + 0.1*obj_normal, max_dist);
            //f_color = vec4(vec3((1.0 + obj_in_way) / 2.0), 1.0);
            //f_color = vec4(sphere.color, 1.0);
            
            if (obj_in_way != -1){
                // In shadow
                f_color = vec4(0.0, 0.0, 0.0, 0.0); 
                return;
            }
            if (object_hit == 0) {
                // Sphere
                f_color = vec4(shade(ray, p_hit, obj_normal, sphere.color, sphere.shininess), 1.0);
               return;
            } else if(object_hit == 1){
                f_color = vec4(shade(ray, p_hit, obj_normal, plane.color, plane.shininess), 1.0);
                return;
            } else {
                // Error
                f_color = vec4(0.8706, 0.9333, 0.0157, 1.0);
            }
           // f_color = vec4(vec3((1.0 + obj_in_way) / 2.0), 1.0);

        //}
    //}
    

}


void marchRay(out int object_hit, inout vec4 ray, in vec3 ray_start, float max_dist){
    
    object_hit = -1;
    for(int i = 0; i < march_iterations; i++) {
        // TODO: Loop over objects
        float signed_dist = max_dist;

        float sphere_signed_dist = sdfSphere(sphere, ray,ray_start);
        float plane_signed_dist = sdfPlane(plane, ray, ray_start);

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
        } else if (signed_dist >= max_dist) {
            object_hit = -1; 
            return;
        } else { 
            ray.w = ray.w + signed_dist;
        }
    }
    return;
    
}


vec3 getNormal(vec3 p, int object_hit){
    if(object_hit == -1) {return vec3(0.);}
    if(object_hit == 0){
        // Sphere
        return normalize(vec3(p - sphere.center));
    } else if (object_hit == 1) {
        // PLane
        return plane.normal;
    } else {
        // Error
        return vec3(0.);
    }
}


float sdfPlane(Plane plane, vec4 ray, vec3 ray_start){
    if(dot(ray.xyz , plane.normal) < 0){
        // plane facing camera
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        vec3 p_plane = plane.normal * -plane.distance;
        return dot(ray_start + ray.xyz * ray.w - p_plane, plane.normal) + plane.distance; 
    } else {
        return maxDistance;
    }
}
float sdfSphere(Sphere sph, vec4 ray, vec3 ray_start){
    return length(ray.xyz * ray.w + ray_start - sph.center) - sph.radius;
}

// Based on general structure of Dr. TJ Jankun-Kelly's Observable Notes: https://observablehq.com/@infowantstobeseen/basic-ray-marching
vec3 shade(vec4 original_ray, vec3 hit_point, vec3 normal, vec3 object_color, float obj_shininess) {
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
        vec3 specular_color =  light_color * pow(e_dot_r, obj_shininess);
    
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
        //return vec3(e_dot_r);
        vec3 specular_color = 0.00001 * dir_light_color * pow(e_dot_r, obj_shininess);
    
        return vec3(amb_color + diffuse_color + specular_color);// 
    }
}


// from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}
