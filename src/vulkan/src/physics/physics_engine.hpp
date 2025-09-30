#pragma once

#include <btBulletDynamicsCommon.h>
#include <unordered_map>
#include <memory>
#include <vector>

#include <glm/gtc/quaternion.hpp>

namespace lve{
  class LveGameObject;
  class PhysicsEngine{
  public:
    PhysicsEngine();
    ~PhysicsEngine();

    void stepSimulation(float deltaTime);

    void addBoxRigidBody(LveGameObject& obj, float mass);
    void removeRigidBody(LveGameObject& obj);
    void syncTransforms(const std::vector<LveGameObject *> &objects); // Bullet → GameObject

    btDiscreteDynamicsWorld *getWorld() { return dynamicsWorld.get(); }

  private:
    struct PhysicsBody {
      btRigidBody* rigidBody{};
      btCollisionShape* shape{};
      btDefaultMotionState* motionState{};
    };

    std::unordered_map<unsigned int, PhysicsBody> rigidBodies;

    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;
    std::unique_ptr<btDiscreteDynamicsWorld> dynamicsWorld;
  };
} // namespcae lve