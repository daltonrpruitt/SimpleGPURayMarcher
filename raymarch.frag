#version 430

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
    float reflectiveness;
};

struct VoxelSDFInfo {
    int id;
    vec3 position;
    vec3 rotation;
    float scale;
    vec3 color;
    float shininess;
    float reflectiveness;
};
 
//Plane information
vec3 plane_norm = normalize(vec3(0., 1, 0));
float plane_dist = 1.5;
vec3 plane_color = vec3(0.2431, 0.7451, 0.0431);
float plane_shininess = 16.0;
float plane_reflectiveness = 0.3;

Plane plane = Plane(plane_norm, plane_dist, plane_color, plane_shininess, plane_reflectiveness);

// Box Info
uniform vec3 box_center = vec3(-2,  0., 8.);
vec3 box_dimensions = vec3(1);
vec3 box_color = vec3(0.9686, 0.9843, 0.0118);
float box_shininess = 32.0;
uniform vec3 box_rotation;



uniform float width;
uniform float height;
// uniform float time;
uniform Sphere sphere;
uniform vec4 back_color;
uniform vec4 light;
uniform vec3 light_color;
bool point_light_shadow = false;

uniform bool using_point_light;
uniform bool using_dir_light;


out vec4 f_color;


int march_iterations = 1024;
uniform vec3 cam_pos; // = vec3(0.0, 0.0, -10.0);
float ambient_coeff = 0.1;
float maxDistance = 1.0e3;
vec3 dir_light = normalize(vec3(1, -1, 1));
vec3 dir_light_color = vec3(1);
bool dir_light_shadow = false;


// Object Voxel SDF Array : Dim = (resolution + 2)^2 *(resolution + 2)    For later: https://community.khronos.org/t/dynamic-array-of-uniforms/63246/2
uniform sampler3D crate_sdf_texture;
uniform float crate_scale;
uniform vec3 crate_rotation;

VoxelSDFInfo crateSDF = VoxelSDFInfo(0, vec3(-0.5, 1, 3), vec3(0), crate_scale, vec3(0.902, 0.5412, 0.0706), 4.0, 0.0);


int numObjects = 3;
int findMinInArray(float[3]);

float rand(vec2);
vec3 shade(vec4, vec3, vec3, vec3, float, bool[2]);
float sdfSphere(Sphere, vec4, vec3);
float sdfPlane(Plane, vec3);
float sdfBox(vec3, vec3, vec3);

void marchRay(out int, inout vec4, in vec3, float);
vec4 reflectedRayMarchColor(vec4, vec3 );
vec3 getNormal(vec3, int);

mat4 rotationX(in float);
mat4 rotationY(in float);
mat4 rotationZ(in float);
mat4 translateFromVec3(in vec3);

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
            float max_dist; 

            float numLights = 0;

            bool in_shadows[2];
            in_shadows[0] = false;
            in_shadows[1] = false;

            if(using_point_light){
                numLights = numLights + 1;
                to_light = normalize(light.xyz - p_hit);
                max_dist = length(light.xyz - p_hit)+1;
                vec4 shadow_ray = vec4(to_light, 0.001);
                int obj_in_way;
                marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, max_dist);
                if (obj_in_way != -1){
                // In shadow
                    //in_shadows = in_shadows + 1;
                    in_shadows[0] = true;
                }

            }
            if (using_dir_light) {
                numLights = numLights + 1;
                to_light = -1.0 * dir_light;
                max_dist = maxDistance;
                vec4 shadow_ray = vec4(to_light, 0.001);
                int obj_in_way;
                marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, max_dist);
                if (obj_in_way != -1){
                    // In shadow            
                    in_shadows[1] = true;
                    //in_shadows = in_shadows + 1;
                    //dir_light_shadow = true;
                }
            }
            if(numLights == 0) { f_color = vec4(0.0); return;}
            //float shadow_fraction = in_shadows/numLights;
            

            if (object_hit == 0) {
                // Sphere
                f_color = vec4(shade(ray, p_hit, obj_normal, sphere.color, sphere.shininess, in_shadows), 1.0);
               return;
            } else if (object_hit == 1){
                // Plane
                vec4 reflection_color = vec4(0);
                if(plane.reflectiveness > 0){
                    vec4 r = vec4(reflect(ray.xyz, obj_normal), 0.0001);
                    reflection_color = reflectedRayMarchColor(r, p_hit + r.w*r.xyz);
                }
                f_color = plane.reflectiveness * reflection_color + 
                        (1.0-plane.reflectiveness) * vec4(shade(ray, p_hit, obj_normal, plane.color, plane.shininess, in_shadows), 1.0);
                // should change something around here to get into the sampler3D
                // f_color = vec4(abs(texture(crate_sdf_texture, gl_FragCoord.xyz/height - 0.2)));//trunc((gl_FragCoord.xy - width/2)/width * crate_scale),0))));
                return;
            } else if (object_hit == 2){
                // Box
                //f_color = vec4(abs(obj_normal), 1.0);return;//Debug
                f_color = vec4(shade(ray, p_hit, obj_normal, box_color, box_shininess, in_shadows), 1.0);
                return;
            } else { 
                // Error
                f_color = vec4(0.9333, 0.0157, 0.0157, 1.0);
            }
           // f_color = vec4(vec3((1.0 + obj_in_way) / 2.0), 1.0);

        //}
    //}
    

}

vec4 reflectedRayMarchColor(vec4 reflected_ray, vec3 reflection_origin){
    int object_hit;
    marchRay(object_hit, reflected_ray, reflection_origin, maxDistance);

    if(object_hit == -1) {
        // Hit nothing
        return back_color; 
    } 

    vec3 p_hit = reflection_origin + reflected_ray.xyz*reflected_ray.w;
    vec3 obj_normal = getNormal(p_hit, object_hit);
    //f_color = vec4(obj_normal*vec3(2, 2, 3), 1); 
    vec3 to_light;
    float max_dist = maxDistance; 
    float numLights = 0;
    //float in_shadows = 0;
    bool in_shadows[2];
    in_shadows[0] = false;
    in_shadows[1] = false;

    if(using_point_light){
        numLights = numLights + 1;
        to_light = normalize(light.xyz - p_hit);
        max_dist = length(light.xyz - p_hit);
        vec4 shadow_ray = vec4(to_light, 0.001);
        int obj_in_way;
        marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, max_dist);
        if (obj_in_way != -1){
        // In shadow
            in_shadows[0] = true;
            //in_shadows = in_shadows + 1;
        }

    }
    if (using_dir_light) {
        numLights = numLights + 1;
        to_light = -1.0 * dir_light;
        max_dist = maxDistance;
        vec4 shadow_ray = vec4(to_light, 0.001);
        int obj_in_way;
        marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, max_dist);
        if (obj_in_way != -1){
        // In shadow
            in_shadows[1] = true;
            //in_shadows = in_shadows + 1;
        }
    }


    //float reflected_shadow_fraction = in_shadows/numLights;
    
    if(in_shadows[0] && in_shadows[1]) {
        return vec4(0);
    }
    if (object_hit == 0) {
        // Sphere
        return vec4(shade(reflected_ray, reflection_origin, obj_normal, sphere.color, sphere.shininess, in_shadows), 1.0);
    } else if (object_hit == 1){
        // Plane
        return vec4(shade(reflected_ray, reflection_origin, obj_normal, plane.color, plane.shininess, in_shadows), 1.0);
    } else if (object_hit == 2){
        // Box
        //f_color = vec4(abs(obj_normal), 1.0);return;//Debug
        return vec4(shade(reflected_ray, reflection_origin, obj_normal, box_color, box_shininess, in_shadows), 1.0);
    } else { 
        // Error
        return vec4(0.9333, 0.0157, 0.0157, 1.0);
    }
}

void marchRay(out int object_hit, inout vec4 ray, in vec3 ray_start, float max_dist){
    
    object_hit = -1;
    for(int i = 0; i < march_iterations; i++) {
        // TODO: Loop over objects
        float signed_dist = max_dist;

        float sphere_signed_dist = sdfSphere(sphere, ray, ray_start);
        float plane_signed_dist = sdfPlane(plane, ray.xyz * ray.w + ray_start);
        float box_signed_dist = sdfBox(box_dimensions, box_center, ray.xyz * ray.w + ray_start);
        
        float distances[3];
        distances[0] = sphere_signed_dist;
        distances[1] = plane_signed_dist;
        distances[2] = box_signed_dist;
        int obj_idx = findMinInArray(distances);

        int object_closer = -1;
        if (distances[obj_idx] < signed_dist) {
            signed_dist = distances[obj_idx];
        }

        if(signed_dist < 0.0001 ){
            object_hit = obj_idx;
            return;
        } else if (signed_dist >= max_dist) {
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
    } else if (object_hit == 2) {
        // Box - based on https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
        // Change to evaluating the gradient of the sdf at 6 offset positions
        //vec3 q = (vec4(p, 1.0)*translateFromVec3(-1.*box_center)*rotationY(box_rotation.y)).xyz - box_dimensions; // change to not global...
        mat4 rotationMat = rotationY(box_rotation.y);
        vec3 q = (vec4(p, 1.0)*translateFromVec3(-1.*box_center)*rotationMat).xyz;
        if ( abs(abs(q.x) - box_dimensions.x) <= 0.001) {
            vec3 pos_x_norm = (vec4(1, 0, 0, 0)*transpose(rotationMat)).xyz;
            if (q.x > 0) {return pos_x_norm;}
            else  {return -1.*pos_x_norm;}
        }
        else if ( abs(abs(q.y) - box_dimensions.y) <= 0.001) {
            vec3 pos_y_norm = (vec4(0, 1, 0, 0)*transpose(rotationMat)).xyz;
            if (q.y > 0) {return pos_y_norm;}
            else  {return -1.*pos_y_norm;}
        }
        else {
            vec3 pos_z_norm = (vec4(0, 0, 1, 0)*transpose(rotationMat)).xyz;
            if (q.z > 0) {return pos_z_norm;}
            else  {return -1.*pos_z_norm;}
        }

        //return normalize((vec4(max(q,0.0) + min(max(q.x,max(q.y,q.z)),0.0), 1.0)*rotationY(-box_rotation.y)).xyz);
        } else {
        // Error
        return vec3(0.);
    }
}


float sdfPlane(Plane plane, vec3 point){
    if(dot(point, plane.normal) > -plane.distance){
        // plane facing point
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        //vec3 p_plane = plane.normal * -plane.distance;
        return dot(point, plane.normal) + plane.distance; 
    } else {
        return maxDistance;
    }
}
float sdfSphere(Sphere sph, vec4 ray, vec3 ray_start){
    return length(ray.xyz * ray.w + ray_start - sph.center) - sph.radius;
}

// Based on https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdfBox(vec3 dimensions, vec3 center, vec3 point) {
    // translate box
  vec3 q = abs(vec4(point, 1.0)*translateFromVec3(-1.*box_center)*rotationY(box_rotation.y)).xyz - dimensions;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdfVoxelSDF(VoxelSDFInfo voxSDF, vec3 point) {
    // Test if in Box shell first
    // Probably some bugs in here
    float shell_sdf = sdfBox(vec3(voxSDF.scale), voxSDF.position, point);
    if(shell_sdf > 0.05) { return shell_sdf;}
    
    // Get sdf from index inside
    float sdf; 
    mat4 rotationMat = rotationX(voxSDF.rotation.x)*rotationY(voxSDF.rotation.y)*rotationZ(voxSDF.rotation.z);

    vec3 sample_pos = ((vec4(point - voxSDF.position, 1.0)*rotationMat).xyz + vec3(voxSDF.scale/2))/voxSDF.scale;
    if(voxSDF.id == 0) {
        sdf = texture(crate_sdf_texture, sample_pos).x;
    } else {
        return maxDistance;
    }
    return sdf;
}

// Based on general structure of Dr. TJ Jankun-Kelly's Observable Notes: https://observablehq.com/@infowantstobeseen/basic-ray-marching
vec3 shade(vec4 original_ray, vec3 hit_point, vec3 normal, vec3 object_color, float obj_shininess, bool in_shadow[2]) {
    vec3 color = ambient_coeff * object_color;

    if(using_point_light){

        if (!in_shadow[0]){
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
        color = color + diffuse_color + specular_color; //* (1-shadow_fraction);
        }
    }
    if(using_dir_light) {
        if(!in_shadow[1]){
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
        vec3 specular_color = dir_light_color * pow(e_dot_r, obj_shininess);
        color = color +  diffuse_color + specular_color ;//* (1-shadow_fraction); 
        }
    }
    return clamp(color , vec3(0.), vec3(1.));
}


// Non-sorted array of floats
int findMinInArray(float distances[3]){
    int min_idx = 0;
    float minimum = distances[0];
    for(int i = 1; i < distances.length(); i++ ) {
        if(distances[i] < minimum) {
            min_idx = i;
            minimum = distances[i];
        }
    }
    return min_idx;
}

// from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

// #################################################################
// https://gist.github.com/onedayitwillmake/3288507
mat4 rotationX( in float angle ) {
	return mat4(	1.0,		0,			0,			0,
			 		0, 	cos(angle),	-sin(angle),		0,
					0, 	sin(angle),	 cos(angle),		0,
					0, 			0,			  0, 		1);
}

mat4 rotationY( in float angle ) {
	return mat4(	cos(angle),		0,		sin(angle),	0,
			 				0,		1.0,			 0,	0,
					-sin(angle),	0,		cos(angle),	0,
							0, 		0,				0,	1);
}

mat4 rotationZ( in float angle ) {
	return mat4(	cos(angle),		-sin(angle),	0,	0,
			 		sin(angle),		cos(angle),		0,	0,
							0,				0,		1,	0,
							0,				0,		0,	1);
}
// #################################################################

mat4 translateFromVec3(in vec3 offset) {
    	return mat4(1,	0,	0,	offset.x,
			 		0,	1,	0,	offset.y,
					0,	0,	1,	offset.z,
					0,	0,	0,	    1);
}
