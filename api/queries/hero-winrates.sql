SELECT ARRAY_TO_JSON(ARRAY(
SELECT
  JSONB_BUILD_OBJECT(
  'actor', actors.actor,
  'winrate', (SELECT
    SUM((CASE WHEN (participant.data->'attributes'->'stats'->>'winner')='true' THEN 1 ELSE 0 END))::float
    /
    COUNT(*)::float
    FROM participant
    WHERE (participant.data->'attributes'->>'actor') = actors.actor)
  )
FROM
(SELECT
  DISTINCT participant.data->'attributes'->>'actor'
  AS actor
  FROM participant)
AS actors)
) AS data;