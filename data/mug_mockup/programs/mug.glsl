#version 330

#if defined VERTEX_SHADER

uniform mat4 Mvp;

in vec3 in_vert;
in vec3 in_norm;

out vec3 v_vert;
out vec3 v_norm;

void main() {
    gl_Position = Mvp * vec4(in_vert, 1.0);
    v_vert = in_vert;
    v_norm = in_norm;
}

#elif defined FRAGMENT_SHADER

uniform vec3 Light;

in vec3 v_vert;
in vec3 v_norm;

out vec4 f_color;

void main() {
    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
    vec3 base = vec3(0.5, 0.5, 0.5) * lum;
    vec3 spec = vec3(1.0, 1.0, 1.0) * pow(lum, 5.7);
    f_color = vec4(base * 0.1 + vec3(10.0 / 256.0) * lum + spec, 1.0);
}

#endif
