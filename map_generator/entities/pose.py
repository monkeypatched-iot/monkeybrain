
class Pose:
    def save_keyframe(self, tx, keyframe):
        """Saves a keyframe to the Neo4j database."""
        keyframe.pose = keyframe.pose[0].tolist()

        keyframe_data = {
            "id": keyframe.id,
            "pose": keyframe.pose
        }

        query = (
                "CREATE (kf:Keyframe {id: "+ str(keyframe_data["id"]) +", pose: $pose})"
              )

        tx.run(query, **keyframe_data)

    def add_unique_constraint(self,tx):
        # Cypher command to add a unique constraint
        query = """
          CREATE CONSTRAINT unique_keyframe_id IF NOT EXISTS
          FOR (k:Keyframe)
          REQUIRE k.id IS UNIQUE
          """
        tx.run(query)
        return True


    def add_relationship(self, tx, kf1_id, kf2_id):
        query = (
            "MATCH (kf1:Keyframe {id: $kf1_id}), (kf2:Keyframe {id: $kf2_id}) "
            "CREATE (kf1)-[:CONNECTED_TO]->(kf2)"
        )
        tx.run(query, kf1_id=kf1_id, kf2_id=kf2_id)

    def node_exists(self,keyframe_id,driver):
      with driver.session() as session:
          result = session.run(
              "MATCH (n:Keyframe {id: $id}) RETURN count(n) > 0 AS node_exists",
              id=keyframe_id
          )
          return result.single()["node_exists"]
