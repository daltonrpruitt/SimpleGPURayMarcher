#version 330

#if defined VERTEX_SHADER

uniform mat4 Mvp;

in vec3 in_vert;
in vec3 in_norm;
in vec2 in_text;

out vec3 v_vert;
out vec3 v_norm;
out vec2 v_text;

void main() {
    gl_Position = Mvp * vec4(in_vert, 1.0);
    v_vert = in_vert;
    v_norm = in_norm;
    v_text = in_text;
}

#elif defined FRAGMENT_SHADER

uniform vec3 Light;
uniform sampler2D Texture;

in vec3 v_vert;
in vec3 v_norm;
in vec2 v_text;

out vec4 f_color;

void main() {
    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
    vec3 base = vec3(0.5, 0.5, 0.5) * lum;
    vec3 spec = vec3(1.0, 1.0, 1.0) * pow(lum, 5.7);
    vec4 tex = texture(Texture, v_text);
    f_color = vec4(base * 0.1 + tex.rgb * lum + spec, tex.a);
}

#endif
