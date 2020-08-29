#version 330

#if defined VERTEX_SHADER
in vec2 in_vert;
out vec2 v_vert;

void main() {
    gl_Position = vec4(in_vert * 2.0 - 1.0, 0.0, 1.0);
    v_vert = in_vert;
}

#elif defined FRAGMENT_SHADER

uniform sampler2D Texture;

in vec2 v_vert;

out vec4 f_color;

void main() {
    f_color = texture(Texture, v_vert);
}

#endif
