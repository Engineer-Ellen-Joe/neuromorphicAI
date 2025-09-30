#version 450

layout (location = 0) in vec3 fragColor;
layout (location = 1) in vec3 fragPosWorld;
layout (location = 2) in vec3 fragNormalWorld;
layout (location = 3) in vec2 fragUv;

layout (location = 0) out vec4 outColor;

struct PointLight {
  vec4 position; // ignore w
  vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo {
  mat4 projection;
  mat4 view;
  mat4 invView;
  vec4 ambientLightColor; // w is intensity
  PointLight pointLights[10];
  int numLights;
} ubo;

layout (set = 1, binding = 1) uniform sampler2D diffuseMap;

layout(push_constant) uniform Push {
  mat4 modelMatrix;
  mat4 normalMatrix;
  int useTexture; // 0 = 꺼짐, 1 = 스프라이트 형식의 텍스쳐(애니메이션 활성화), 2 = 스프라이트 형식이 아닌 텍스쳐
  int currentFrame;
  int objectState;
  int direction;
} push;

void main() {
  vec3 diffuseLight = ubo.ambientLightColor.xyz * ubo.ambientLightColor.w;
  vec3 specularLight = vec3(0.0);
  vec3 surfaceNormal = normalize(fragNormalWorld);

  vec3 cameraPosWorld = ubo.invView[3].xyz;
  vec3 viewDirection = normalize(cameraPosWorld - fragPosWorld);

  for (int i = 0; i < ubo.numLights; i++) {
    PointLight light = ubo.pointLights[i];
    vec3 directionToLight = light.position.xyz - fragPosWorld;
    float attenuation = 1.0 / dot(directionToLight, directionToLight); // distance squared
    directionToLight = normalize(directionToLight);

    float cosAngIncidence = max(dot(surfaceNormal, directionToLight), 0);
    vec3 intensity = light.color.xyz * light.color.w * attenuation;

    diffuseLight += intensity * cosAngIncidence;

    // specular lighting
    vec3 halfAngle = normalize(directionToLight + viewDirection);
    float blinnTerm = dot(surfaceNormal, halfAngle);
    blinnTerm = clamp(blinnTerm, 0, 1);
    blinnTerm = pow(blinnTerm, 512.0); // higher values -> sharper highlight
    specularLight += intensity * blinnTerm;
  }

  // sprite texture animation
  vec2 uvOffset = vec2(0.0);
  float frameWidth = 1.0 / 6.0; // 기본 가로 프레임 개수
  if(push.objectState == 1){
    frameWidth = 1.0;
  }
  float frameHeight = 1.0 / 2.0; // 세로 프레임 개수
  uvOffset.x = frameWidth * float(push.currentFrame);
  uvOffset.y = frameHeight * float(push.objectState);
  vec2 animatedUv = fragUv + uvOffset;
  if(push.direction == 2) { // 방향이 LEFT일 경우 UV 좌표 반전
    animatedUv.x = 1.0 - animatedUv.x;
  }

  vec3 color = fragColor;
  // 임시로 오브젝트의 vec3과 텍스쳐의 vec4를 분리해놓음 (이후 통합할 예정)
  vec4 sampledColor = texture(diffuseMap, animatedUv);
  if (push.useTexture == 1) {
    color = sampledColor.xyz;
  }
  
  outColor = vec4(diffuseLight * color + specularLight * fragColor, 1.0);

  // 텍스쳐에 투명도가 있을경우 렌더링을 하지 않음
  if(sampledColor.a < 0.1){
    discard;
  }
}
