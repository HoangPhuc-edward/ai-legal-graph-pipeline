CREATE CONSTRAINT norm_id IF NOT EXISTS FOR (n:Norm) REQUIRE n.norm_id IS UNIQUE;
CREATE CONSTRAINT comp_id IF NOT EXISTS FOR (c:Component) REQUIRE c.comp_id IS UNIQUE;
CREATE CONSTRAINT unit_id IF NOT EXISTS FOR (t:TextUnit) REQUIRE t.unit_id IS UNIQUE;
CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.action_id IS UNIQUE;

CREATE INDEX comp_level IF NOT EXISTS FOR (c:Component) ON (c.level);
CREATE INDEX norm_status IF NOT EXISTS FOR (n:Norm) ON (n.validity_status);

CREATE VECTOR INDEX textunit_embedding_index IF NOT EXISTS
FOR (t:TextUnit) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
