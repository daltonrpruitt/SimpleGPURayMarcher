#version 430

struct MaterialProperties {
    vec3 color;
    float shininess;
    float reflectiveness;
    bool is_transparent;
    float glossiness;
};


struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float shininess;
    float reflectiveness;
    bool is_transparent;
    float glossiness;
};

struct Light {
    vec3 point;
    vec3 color;
    float intensity;
};

struct SphereLight {
    vec3 center;
    float radius;
    vec3 color;
    float intensity;
};

// SphereLight volLight = SphereLight(vec3(1, 5, 2), 2., vec3(1), 1);
uniform SphereLight volLight;




struct Plane {
    vec3 normal;
    float distance;
    vec3 color;
    float shininess;
    float reflectiveness;
    float glossiness;
};

struct VoxelSDFInfo {
    int id;
    vec3 position;
    vec3 rotation;
    float scale;
    vec3 color;
    float shininess;
    float reflectiveness;
    bool is_transparent;
};
// Object Voxel SDF Array : Dim = (resolution + 2)^2 *(resolution + 2)    For later: https://community.khronos.org/t/dynamic-array-of-uniforms/63246/2
uniform vec3 crate_center;
uniform vec3 crate_rotation;
uniform float crate_scale;

VoxelSDFInfo crateSDFInfo = VoxelSDFInfo(0, crate_center, crate_rotation, crate_scale, vec3(0.7922, 0.549, 0.2275), 8.0,  0.0, false);
layout(binding = 0) uniform sampler3D crate_sdf_texture;

uniform VoxelSDFInfo linkSDFInfo;
layout(binding = 1) uniform sampler3D link_sdf_texture;
//vec3 temp_color = linkSDFInfo.color;

//Plane plane = Plane(plane_norm, plane_dist, plane_color, plane_shininess, plane_reflectiveness, plane_glossiness);
uniform Plane plane;
//plane.normal = normalize(plane.normal);

// Box Info
uniform vec3 box_center; // = vec3(-2,  0., 8.);
vec3 box_dimensions = vec3(1);
vec3 box_color = vec3(0.9686, 0.9843, 0.0118);
float box_shininess = 32.0;
float box_reflectiveness = 0.0;
uniform vec3 box_rotation;

struct Box {
    vec3 center;
    vec3 dimensions;
    vec3 color;
    float shininess;
    float reflectiveness;

};

Box box = Box(box_center, box_dimensions, box_color, box_shininess, box_reflectiveness);

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
uniform bool using_sphere_light;


out vec4 f_color;


int march_iterations = 1024;
uniform int max_recursive_depth;
uniform vec3 cam_pos; // = vec3(0.0, 0.0, -10.0);
uniform int antialiasing_sample_frequency;
uniform bool use_depth_of_field;
uniform float u_focal_distance;
uniform float u_lens_distance;
uniform float u_lens_radius;
uniform int u_dof_samples;

uniform float u_gloss_blur_coeff;
uniform int u_gloss_samples;




float ambient_coeff = 0.1;
float maxDistance = 1.0e3;
vec3 dir_light = normalize(vec3(1, -1, 1));
vec3 dir_light_color = vec3(1);
bool dir_light_shadow = false;

//http://www.pbr-book.org/3ed-2018/Utilities/Main_Include_File.html
const float Pi      = 3.14159265358979323846;
const float InvPi   = 0.31830988618379067154;
const float Inv2Pi  = 0.15915494309189533577;
const float Inv4Pi  = 0.07957747154594766788;
const float PiOver2 = 1.57079632679489661923;
const float PiOver4 = 0.78539816339744830961;
const float Sqrt2   = 1.41421356237309504880;

#define NUM_OBJECTS 5
int numObjects = NUM_OBJECTS;
int findMinInArray(float[NUM_OBJECTS]);

float rand(vec2);
vec3 shade(vec4, vec3, vec3, vec3, float, float[3]);
float sdfSphere(Sphere, vec3);
float sdfPlane(Plane, vec3);
float sdfBox(vec3, vec3, vec3, vec3);
float sdfVoxelSDF(VoxelSDFInfo, vec3);

void marchRay(out int, inout vec4, in vec3, float, bool);
vec4 iterativeDepthMarchRay(inout vec4, in vec3, in float);
vec4 refractOnObject(vec4, vec3, vec3, int);

vec3 getNormal(vec3, int);
void check_shadows(in vec3, in vec3, in out float[3]);
MaterialProperties getMaterialProperties(int obj_hit);


mat4 rotationX(in float);
mat4 rotationY(in float);
mat4 rotationZ(in float);
mat4 translateFromVec3(in vec3);
mat4 rotationFromVec3(in vec3);
mat4 invRotationFromVec3(in vec3);

//http://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models.html#fragment-ProjectiveCameraProtectedData-1
void thinLensRay(inout vec4 original_ray, inout vec3 ray_origin, vec2 jitter){
    
    if(u_lens_radius > 0){
        // Sample point on lens
        
        vec2 point = original_ray.xy * (-cam_pos.z/original_ray.z); // I see issues with this line in the future
        point = vec2(rand(point*13+jitter*7), rand(point*39 + jitter*3));
        vec2 diskPt;

        if (point.x == 0 && point.y == 0) { 
            diskPt = vec2(0);
        } else {
            float r ;
            float theta;
            if (abs(point.x) > abs(point.y)) {
                r = point.x;
                theta = PiOver4 *(point.y/point.x);
            } else {     
                r = point.y;
                theta = PiOver2 - PiOver4 *(point.x/point.y);
            }
            diskPt = r * vec2(cos(theta), sin(theta));
        }

        diskPt *= u_lens_radius;

        // Compute point on plane of focus
        float ft = u_focal_distance/(original_ray.z);
        vec3 pFocus = original_ray.xyz * ft + ray_origin;;

        // update ray with lens effect
        ray_origin = vec3(diskPt.x, diskPt.y, ray_origin.z);
        original_ray = vec4(normalize(pFocus - ray_origin), original_ray.w);
    }
}



void main() {

    vec2 window_size = vec2(width,height);
    //f_color = ray;
    
    f_color = back_color;
    vec4 color = vec4(0,0,0,0);
    
    // From page 310 of Shirley and Marschner
    int sample_frequency = antialiasing_sample_frequency; // 
    for(int p = 0; p < sample_frequency; p++) {  
        for(int q = 0; q < sample_frequency; q++) {  
            float f_samp_freq = float(sample_frequency);
           
            // Make rays offset uniform amounts from center (pg 310)
            vec2 sub_region = vec2((p+0.5)/f_samp_freq - 0.5, (q+0.5)/f_samp_freq - 0.5);
            vec2 random_shift = vec2(rand(gl_FragCoord.xy + 2*sub_region),rand(vec2(gl_FragCoord.xy + sub_region)))/f_samp_freq;
            
            vec2 sub_pixel = gl_FragCoord.xy + sub_region + vec2(random_shift)/f_samp_freq;
            
            vec3 ray_origin = cam_pos;
            vec4 ray = vec4(normalize(vec3((sub_pixel - window_size/2.0)/height*abs(cam_pos.z),-cam_pos.z)-cam_pos), 0.001);
            
            vec4 dof_ray;
            vec3 dof_ray_start;
            vec4 dof_color = vec4(0);

            if (use_depth_of_field) {
                for(int i=0; i< u_dof_samples; i++){
                    dof_ray = ray;
                    dof_ray_start = ray_origin;
                    //( original_ray,  ray_origin,  focal_distance, lens_distance, lens_radius)
                    float fi = float(i);
                    vec2 jitter = vec2(fi * Pi + sub_region.y*13, fi*17 + sub_region.x * height);                
                    thinLensRay( dof_ray, dof_ray_start, jitter);
                    dof_color += iterativeDepthMarchRay(dof_ray, dof_ray_start, maxDistance);
                }
                float samples = float(u_dof_samples);
                color += vec4((dof_color/samples).rgb, 1.0);
            } else {
            // I would really like to make a centralized "object" struct, but oh well ¯\_(ツ)_/¯
                color += iterativeDepthMarchRay(ray, ray_origin, maxDistance);
            }
            
            

        }
    }
    f_color = vec4((color/pow(sample_frequency,2.0)).rgb, 1.0);
    

}


void marchRay(out int object_hit, inout vec4 ray, in vec3 ray_start, float max_dist){
    
    object_hit = -1;
    for(int i = 0; i < march_iterations; i++) {
        // TODO: Loop over objects
        float signed_dist = max_dist;

        float sphere_signed_dist = sdfSphere(sphere, ray.xyz * ray.w + ray_start);
        float plane_signed_dist = sdfPlane(plane, ray.xyz * ray.w + ray_start);
        float box_signed_dist = sdfBox(box_dimensions, box_center, box_rotation, ray.xyz * ray.w + ray_start);
        float crate_signed_dist = sdfVoxelSDF(crateSDFInfo , ray.xyz * ray.w + ray_start);
        float link_signed_dist = sdfVoxelSDF(linkSDFInfo , ray.xyz * ray.w + ray_start);
        
        float distances[NUM_OBJECTS];
        distances[0] = sphere_signed_dist;
        distances[1] = plane_signed_dist;
        distances[2] = box_signed_dist;
        distances[3] = crate_signed_dist;
        distances[4] = link_signed_dist;
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

void marchRefractedRay(int object_inside, inout vec4 refracted_ray, in vec3 ray_start){
    
    for(int i = 0; i < march_iterations; i++) {
        float sdf = -1.0e3;
        float max_dist = 0;
        if(object_inside == 0 ) {
            max_dist = 2 * sphere.radius + 0.5;
            sdf = sdfSphere(sphere, refracted_ray.xyz * refracted_ray.w + ray_start);
        }else {
            refracted_ray.w = -1.0;
            return;
        }

        if(sdf > -0.0001){
            return;
        } else if(refracted_ray.w > max_dist){
            // Does not exit sphere?
            refracted_ray.w = -1;
            return;
        } else { 
            refracted_ray.w += -sdf;
        }

    }
    refracted_ray.w = -1;
    return;

}

// Page 305 of Shirley and Marschner
float calcFresnelSchlickApproximation(vec3 incident, vec3 normal, float eta1, float eta2){
    //return 0.0;
    float cos_between = dot(-incident, normal);
    float R0 = pow( ( eta1 - 1)/(eta2 + 1), 2.0);
    return R0 + (1-R0) * pow((1- cos_between), 5.0);

}



vec4 iterativeDepthMarchRay(inout vec4 ray, in vec3 ray_start, float max_dist){
    
    // This iterative structure from https://www.cs.uaf.edu/2012/spring/cs481/section/0/lecture/02_07_recursion_reflection.html
    vec4 output_color = vec4(0);    
    float color_fraction = 1.0;
    int i = 1;
    MaterialProperties mat_props;
    //int gloss_iterations = 4;

    for( i; i <= max_recursive_depth; i++) {

        if(color_fraction < 1.0){ // Within some reflection
            
            if(mat_props.glossiness > 0 ) { // Last hit object was glossy
                
                // Based off of http://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/drt.pdf
                //New orthonormal bases
                vec3 u = cross(ray.xyz, vec3(0.,0.,1.)); // Ignoring degenerate case for now
                vec3 v = cross(ray.xyz, u);

                vec3 gloss_color = vec3(0);
                for(int i=0; i < u_gloss_samples; i++){
                    float f_i = float(i);
                    vec2 sq_pt = vec2(rand(vec2(24+f_i*13,5+f_i*7)+ray.xy*7),
                                      rand(vec2(17-f_i*43,5-f_i*13)+ray.xy*41)); 
                    /*
                    float sqrt_y = sqrt(sq_pt.y);
                    float _2pix = 2. * 3.14159256 * sq_pt.x;
                    vec2 point_in_circle = vec2(sqrt_y * cos(_2pix), sqrt_y * sin(_2pix)); // Range 
                    */

                    vec2 perturbation = u_gloss_blur_coeff * (-1./2.0 + sq_pt);

                    // Make new ray from old ray
                    vec4 gloss_ray = vec4(normalize(ray.xyz + (perturbation.x * u + perturbation.y * v) * mat_props.glossiness), ray.w);

                    // March new ray
                    int object_hit = -1;
                    marchRay(object_hit, gloss_ray, ray_start, max_dist);
                    if(object_hit == -1) {
                        // Hit nothing; show background color
                        gloss_color += back_color.rgb; 
                        continue; // Can't reflect....
                    } 
                    vec3 p_hit = ray_start + gloss_ray.xyz*gloss_ray.w;
                    vec3 obj_normal = getNormal(p_hit, object_hit);

                    // Shade/Lighting         
                    float lighting_fraction[3]; // TODO use a constant to size...
                    check_shadows(p_hit, obj_normal, lighting_fraction); // Just wanted to separate this part out, honestly;
                    MaterialProperties reflect_mat_props = getMaterialProperties(object_hit);
                    float reflectance = 1.0;

                    if(reflect_mat_props.is_transparent) {
                        reflectance =  calcFresnelSchlickApproximation(gloss_ray.xyz, obj_normal, 1.003, 1.5);
                        
                        if( reflectance < 0.95 ) {
                            vec4 refracted_color = refractOnObject(gloss_ray, p_hit, obj_normal, object_hit );
                            //return refracted_color;
                            if(length(refracted_color - vec4(-1.)) < 0.01) { return vec4(0.8941, 0.9922, 0.0, 1.0);}//Error
                            gloss_color += color_fraction * (1-reflectance) * refracted_color.xyz;
                        } else {
                            reflectance = 1.0;
                        }
                    }

                    gloss_color += reflectance * shade(gloss_ray, p_hit, obj_normal, reflect_mat_props.color, reflect_mat_props.shininess, lighting_fraction);
                
                }
                float f_gloss_iterations = float(u_gloss_samples);
                output_color += vec4(color_fraction * gloss_color / f_gloss_iterations, 1.);
                color_fraction *= 0.0; // Terminate reflections at 0
                break;
            }
        }


        int object_hit = -1;
        marchRay(object_hit, ray, ray_start, max_dist);
        if(object_hit == -1) {
            // Hit nothing; show background color
            output_color += back_color; 
            break; // Can't reflect....
        } 

        vec3 p_hit = ray_start + ray.xyz*ray.w;
        vec3 obj_normal = getNormal(p_hit, object_hit);

        int numLights = (using_dir_light?1:0) + (using_point_light?1:0) + (using_sphere_light?1:0);


        // Shadows
        if(numLights == 0) { return vec4(0.0);}
        float lighting_fraction[3]; // TODO use a constant to size...

        check_shadows(p_hit, obj_normal, lighting_fraction); // Just wanted to separate this part out, honestly;
        
        
        // Shading
        mat_props = getMaterialProperties(object_hit);
        
        //vec4 refracted_color = vec4(0.);

        float reflectance = 1.0;
        
        if(mat_props.is_transparent) {
            reflectance =  calcFresnelSchlickApproximation(ray.xyz, obj_normal, 1.003, 1.5);
            
            if( reflectance < 0.95 ) {
                vec4 refracted_color = refractOnObject(ray, p_hit, obj_normal, object_hit );
                //return refracted_color;
                if(length(refracted_color - vec4(-1.)) < 0.01) { return vec4(0.8941, 0.9922, 0.0, 1.0);}//Error
                output_color += color_fraction * (1-reflectance) * refracted_color;
            } else {
                reflectance = 1.0;
            }
        }


        output_color += reflectance * color_fraction * vec4(shade(ray, p_hit, obj_normal, mat_props.color, mat_props.shininess, lighting_fraction), 1.0);
        color_fraction *= mat_props.reflectiveness;

        // Multiply by Fresnel, as well..?
        //

        if (color_fraction < 0.05 ){ break; } // Not much effect left


        // setup for reflected iteration
        ray = vec4(reflect(ray.xyz, obj_normal), 0.001);
        ray_start = p_hit;

        // TODO: Transparency
        // Shoot transmission rays here

    }
    float fi = float(i);
    return output_color / fi;
}



vec4 refractOnObject(vec4 ray, vec3 ray_start, vec3 obj_normal, int object_inside){
    float eta = 1.003/1.5;

    float head_start = 0.01;
    vec4 refracted_ray = vec4(refract(ray.xyz, obj_normal, eta), head_start);
    if(refracted_ray.z < 0) {discard;}




    marchRefractedRay(object_inside, refracted_ray, ray_start);
    if( refracted_ray.w < 0) {return vec4(-1);}
    if(length(refracted_ray.xyz) < 0.01) { return vec4(-1.); }

    vec3 new_ray_start = ray_start + refracted_ray.xyz*refracted_ray.w;
    vec3 new_normal = getNormal(new_ray_start, object_inside);
    vec4 new_ray = vec4(refract(ray.xyz, -new_normal, 1.0/eta), head_start);
    //new_ray.xyz = -new_ray.xyz; // Will be facing wrong way...? Maybe
    //return vec4(new_ray.xy, new_ray.z/2, 1);

    int obj_in_refraction;
    marchRay(obj_in_refraction, new_ray, new_ray_start, maxDistance);
    
    if(obj_in_refraction < 0) {return back_color;} //Hit background
    vec3 new_p_hit = new_ray_start + new_ray.xyz * new_ray.w;
    vec3 lighting_normal = getNormal(new_p_hit, obj_in_refraction);

    int numLights = (using_dir_light?1:0) + (using_point_light?1:0) + (using_sphere_light?1:0);

    // Shadows
    float lighting_fraction[3]; // TODO use a constant to size...
    check_shadows(new_p_hit, lighting_normal, lighting_fraction); 

    MaterialProperties mat_props = getMaterialProperties(obj_in_refraction);

    return vec4(shade(new_ray, new_p_hit, lighting_normal, mat_props.color, mat_props.shininess, lighting_fraction), 1.0);
}


MaterialProperties getMaterialProperties(int obj_hit) {

        if (obj_hit == 0) {
            // Sphere
            return MaterialProperties(sphere.color, sphere.shininess, sphere.reflectiveness, sphere.is_transparent, sphere.glossiness);
        } else if (obj_hit == 1){
            return MaterialProperties(plane.color, plane.shininess, plane.reflectiveness, false, plane.glossiness);
        } else if (obj_hit == 2){
            return MaterialProperties(box.color, box.shininess, box.reflectiveness, false, 0.0);
        } else if (obj_hit == 3){
            return MaterialProperties(crateSDFInfo.color, crateSDFInfo.shininess, crateSDFInfo.reflectiveness, false, 0.0);

        } else if (obj_hit == 4){
            return MaterialProperties(linkSDFInfo.color, linkSDFInfo.shininess, linkSDFInfo.reflectiveness, linkSDFInfo.is_transparent, 0.0);
        } else { 
            // Error
            return MaterialProperties(vec3(-1.), -1., -1., false, -1);
        }

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
        mat4 rotationMat = rotationFromVec3(box_rotation);// rotationY(box_rotation.y);
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
        } else {
            vec3 pos_z_norm = (vec4(0, 0, 1, 0)*transpose(rotationMat)).xyz;
            if (q.z > 0) {return pos_z_norm;}
            else  {return -1.*pos_z_norm;}
        }
        //return normalize((vec4(max(q,0.0) + min(max(q.x,max(q.y,q.z)),0.0), 1.0)*rotationY(-box_rotation.y)).xyz);

    } else if (object_hit == 3 || object_hit == 4) {
        // Crate SDF
        //return vec3(0, 0, -1);
        VoxelSDFInfo currVoxSDF; 
        if(object_hit == 3) {currVoxSDF = crateSDFInfo; }
        else { currVoxSDF = linkSDFInfo;}
        vec3 rot = currVoxSDF.rotation;
        // Store rotation and inverse rotation 

        mat4 rotationMat = rotationFromVec3(rot);
        mat4 inverseRotMat = invRotationFromVec3(rot);
        vec3 sample_pos = ((vec4(p - currVoxSDF.position, 1.0)*rotationMat).xyz - vec3(currVoxSDF.scale))/(currVoxSDF.scale*2);
        float deltDist = 0.03;
        float deltSDFX, deltSDFY, deltSDFZ;
        if(object_hit == 3) { // Crate
            deltSDFX = texture(crate_sdf_texture, sample_pos + vec3(deltDist, 0, 0)).x - texture(crate_sdf_texture, sample_pos - vec3(deltDist, 0,0)).x;
            deltSDFY = texture(crate_sdf_texture, sample_pos + vec3(0, deltDist, 0)).x - texture(crate_sdf_texture, sample_pos -  vec3(0, deltDist, 0)).x;
            deltSDFZ = texture(crate_sdf_texture, sample_pos +  vec3(0, 0, deltDist)).x - texture(crate_sdf_texture, sample_pos -  vec3(0, 0, deltDist)).x;
        } else { // Toon Link
            deltSDFX = texture(link_sdf_texture, sample_pos + vec3(deltDist, 0, 0)).x - texture(link_sdf_texture, sample_pos - vec3(deltDist, 0,0)).x;
            deltSDFY = texture(link_sdf_texture, sample_pos + vec3(0, deltDist, 0)).x - texture(link_sdf_texture, sample_pos -  vec3(0, deltDist, 0)).x;
            deltSDFZ = texture(link_sdf_texture, sample_pos +  vec3(0, 0, deltDist)).x - texture(link_sdf_texture, sample_pos -  vec3(0, 0, deltDist)).x;
        }
        return normalize((vec4(vec3(deltSDFX, deltSDFY, deltSDFZ), 0.)*inverseRotMat).xyz);
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
float sdfSphere(Sphere sph, vec3 point){
    return length(point - sph.center) - sph.radius;
}

// Based on https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdfBox(vec3 dimensions, vec3 center, vec3 rotation, vec3 point) {
    // translate box
  vec3 q = abs(vec4(point - center, 1.0)*rotationFromVec3(rotation)).xyz - dimensions;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdfVoxelSDF(VoxelSDFInfo voxSDF, vec3 point) {
    // Test if in Box shell first
    // Probably some bugs in here
    float shell_sd = sdfBox(vec3(voxSDF.scale), voxSDF.position, voxSDF.rotation, point);
    float shell_offset = 0.0005;
    if(shell_sd > shell_offset ) { return shell_sd;}
    //return 0.000001;

    // Get sdf from index inside
    float sd; 
    mat4 rotationMat = rotationFromVec3(voxSDF.rotation);//rotationX(voxSDF.rotation.x)*rotationY(voxSDF.rotation.y)*rotationZ(voxSDF.rotation.z);


    vec3 sample_pos = ((vec4(point - voxSDF.position, 1.0)*rotationMat).xyz)/(voxSDF.scale*2); 
    
    //if(!all(lessThan(abs(sample_pos), vec3(0.5)))) {
    //    return shell_offset;
    //}
    

    sample_pos += vec3(0.5);
    if(voxSDF.id == 0) {
        sd = texture(crate_sdf_texture, sample_pos).x/(voxSDF.scale*2);
    } else if(voxSDF.id == 1){
        sd = texture(link_sdf_texture, sample_pos).x/(voxSDF.scale*2);
    } else {
        sd = maxDistance; // should show errors real quick (unless dealing with plane...?)
    }
    if(sd < shell_sd) {
         sd = shell_sd; 
    }
    return sd;
}

// Based on general structure of Dr. TJ Jankun-Kelly's Observable Notes: https://observablehq.com/@infowantstobeseen/basic-ray-marching
vec3 shade(vec4 original_ray, vec3 hit_point, vec3 normal, vec3 object_color, float obj_shininess, float lighting_fraction[3]) {
    vec3 color = ambient_coeff * object_color;

    // This if () { if () {    }} structure is due to the incompatibility of uniform bool and temp bool, apparently
    if(using_point_light){ if ( lighting_fraction[0] > 0.) {
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
        color += diffuse_color + specular_color; //* (1-shadow_fraction);
    }}
    if(using_dir_light) { if (lighting_fraction[1] > 0.) { 
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
        color += diffuse_color + specular_color ;//* (1-shadow_fraction); 
    } }
    if(using_sphere_light) { if (lighting_fraction[2] > 0) { 
        vec3 vec_to_light = normalize(volLight.center.xyz - hit_point);
        float lambertian = clamp(dot(vec_to_light, normal), 0.0, 1.0);
        
        vec3 diffuse_color = volLight.color * lambertian * object_color;
                                
        // Reflected Light (Negative because shadow ray pointing away from surface) Shirley & Marschner pg.238
        // Check if is actually reflecting the correct way
        vec3 reflected_vec = reflect(-vec_to_light, normal);
        //Above is effectively normalize(2.0 * dot(vec_to_light, norm) * norm - vec_to_light);
        vec3 e_vec = normalize(-1.0 * original_ray.xyz);  // negative so facing correct way
        float e_dot_r = max(dot(e_vec, reflected_vec), 0.0);
        vec3 specular_color =  light_color * pow(e_dot_r, obj_shininess);
        color += lighting_fraction[2]*(diffuse_color + specular_color); 
    } }

    // maybe divide by number of lights being used before clamping?
    return clamp(color , vec3(0.), vec3(1.));
}


// Non-sorted array of floats
int findMinInArray(float distances[NUM_OBJECTS]){
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

void check_shadows(in vec3 p_hit, in vec3 obj_normal, in out float lighting_fraction[3]){
    // Lighting
    lighting_fraction[0] = 1.;    
    lighting_fraction[1] = 1.;    
    lighting_fraction[2] = 1.;

    vec3 to_light;
    float dist_to_light;

    if(using_point_light){
        to_light = normalize(light.xyz - p_hit);
        dist_to_light = length(light.xyz - p_hit)+1;
        vec4 shadow_ray = vec4(to_light, 0.001);
        int obj_in_way;
        marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, dist_to_light);
        if (obj_in_way != -1){
        // In shadow
            lighting_fraction[0] = 0.;
        }

    }
    if (using_dir_light) {
        to_light = -1.0 * dir_light;
        dist_to_light = maxDistance;
        vec4 shadow_ray = vec4(to_light, 0.001);
        int obj_in_way;
        marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, dist_to_light);
        if (obj_in_way != -1){
            // In shadow            
            lighting_fraction[1] = 0.;
        }
    }
    if (using_sphere_light) {
        to_light = normalize(volLight.center.xyz - p_hit);
        dist_to_light = length(volLight.center.xyz - p_hit)+1;

        //vec3 perp_vec = 
        // 1. Get ratio b/w radius and distance
        float R_D = volLight.radius / dist_to_light; 
        // 2. get two vectors orthogonal to to_light (assuming to_light is not straight down)
        vec3 offV1 = cross(to_light, vec3(0.,0.,1.));
        vec3 offV2 = cross(to_light, offV1);

        // Use these vectors as offsets to the to_light vector with coefficients < R_D
        //   to construct the shadow feeler rays (now we're thinking with portals!)
        // This seems hacky to me, but eh, it'll have to do for now

        int grid_size = 3;
        float f_gs = float(grid_size);
        int num_shadow_feelers = 0;
        int num_hit = 0;
        vec2 coefficients;
        vec4 base_shadow_ray = vec4(to_light, 0.001);
        vec4 shadow_ray;


        for(int i=0; i < grid_size; i++) {
            for (int j=0; j < grid_size; j++) {
                if( i % grid_size == 0 || i % grid_size == grid_size-1 || 
                    j % grid_size == 0 || j % grid_size == grid_size-1   ) { // Only perimeter

                    // Test inside
                    float fi = float(i);
                    float fj = float(j);
                    vec2 sub_region = vec2((fi+0.5) , (fj+0.5))/f_gs;
                    sub_region += vec2(rand(p_hit.xy * sub_region), rand(p_hit.yx * sub_region))/(2.*f_gs);


                    //f_color = vec4(j, k, p_hit.z / 10, 1);return;
                    //float rand1 = rand(p_hit.yz*p_hit.zx + vec2(fi,fj));
                    //float rand2 = rand(p_hit.zx*p_hit.yx + vec2(fi,fj));
                    // From https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using
                    float sqrt_y = sqrt(sub_region.y);
                    float _2pix = 2. * 3.14159256 * sub_region.x;
                    vec2 point_in_circle = vec2(sqrt_y * cos(_2pix), sqrt_y * sin(_2pix))*2. - vec2(1.); //vec2(j,k) / (f_gs/2.) - vec2(0.5);

                    // outside of circle diameter = grid_size
                    //if( length(point_in_circle) > 1.414213562 ) {f_color = vec4(0.9412, 0.0235, 0.0235, 0.0);return; } 
                    
                    // Is inside the circle
                    num_shadow_feelers++;
                    coefficients = R_D * point_in_circle; // Random offset with prime offset within ¯\_(ツ)_/¯ Y not?
                    
                    shadow_ray = base_shadow_ray + vec4(offV1 * coefficients.x + offV2 * coefficients.y,base_shadow_ray.w);
                    int obj_in_way;
                    marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, dist_to_light);
                    if (obj_in_way != -1){
                        // In shadow            
                        num_hit++;
                    }
                }
            }
        }
        // Exit early if all of perimeter hits or doesn't
        if (num_hit == 0 ){lighting_fraction[2] = 1.0;  return;}
        if (num_hit == num_shadow_feelers ){lighting_fraction[2] = 0.0; return;}


        for(int i=1; i < grid_size-1; i++) {
            for (int j=1; j < grid_size-1; j++) {
                float f_gs = float(grid_size);
                float fi = float(i);
                float fj = float(j);
                vec2 sub_region = vec2((fi+0.5) , (fj+0.5))/f_gs;
                sub_region += vec2(rand(p_hit.xy * sub_region), rand(p_hit.yx * sub_region))/(2.*f_gs);

                //f_color = vec4(j, k, p_hit.z / 10, 1);return;

                // From https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using
                float sqrt_y = sqrt(sub_region.y);
                float _2pix = 2. * 3.14159256 * sub_region.x;
                vec2 point_in_circle = vec2(sqrt_y * cos(_2pix), sqrt_y * sin(_2pix))*2. - vec2(1.); //vec2(j,k) / (f_gs/2.) - vec2(0.5);

                // outside of circle diameter = grid_size
                //if( length(point_in_circle) > 1.414213562 ) {f_color = vec4(0.9412, 0.0235, 0.0235, 0.0);return; } 
                
                // Is inside the circle
                num_shadow_feelers++;
                coefficients = R_D * point_in_circle; // Random offset with prime offset within ¯\_(ツ)_/¯ Y not?
                
                shadow_ray = base_shadow_ray + vec4(offV1 * coefficients.x + offV2 * coefficients.y,0.);
                int obj_in_way;
                marchRay(obj_in_way, shadow_ray, p_hit + 0.001*obj_normal, dist_to_light);
                if (obj_in_way != -1){
                    // In shadow            
                    num_hit++;
                }
            }
        }
        float f_num_hit = float(num_hit);
        float f_num_shadow_feelers = float(num_shadow_feelers);

        lighting_fraction[2] = 1 - f_num_hit / f_num_shadow_feelers;

    }
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

mat4 rotationFromVec3(in vec3 rotations) {
    return rotationY(rotations.y)*rotationX(rotations.x)*rotationZ(rotations.z);
}

mat4 invRotationFromVec3(in vec3 rotations) {
    return rotationZ(-rotations.z)*rotationX(-rotations.x)*rotationY(-rotations.y);
}