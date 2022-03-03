#version 330

layout(location=0) in vec3 vert_position;
layout(location=1) in vec2 vert_texture_position;

out vec2 frag_texture_pos;

void main()
{
	gl_Position = vec4(vert_position, 1.f);
	frag_texture_pos = vert_texture_position;
}