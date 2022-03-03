#version 330

uniform sampler2D SourceTexture;

in vec2 frag_texture_pos;

out vec4 result;

void main()
{
	result = texture(SourceTexture, frag_texture_pos);
}