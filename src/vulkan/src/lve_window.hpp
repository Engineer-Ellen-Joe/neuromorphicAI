#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
namespace lve {

  class LveWindow {
  public:
    LveWindow(int w, int h, std::string name); // LveWindow 초기화
    ~LveWindow();                              // LveWindow 자원 해제

    // 복사 생성자, 복사 대입 연산자 삭제 (LveWindow 복사 불가능)
    LveWindow(const LveWindow &) = delete;
    LveWindow &operator=(const LveWindow &) = delete;

    // 이벤트 반환 (창 종료)
    bool shouldClose() { return glfwWindowShouldClose(window); }
    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
    bool wasWindowResized() { return framebufferResized; }
    void resetWindowResizedFlag() { framebufferResized = false; }
    GLFWwindow *getGLFWwindow() const { return window; }

    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);

  private:
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
    void initWindow();

    int width;
    int height;
    bool framebufferResized = false;

    std::string windowName;
    GLFWwindow *window;
  };
} // namespace lve
