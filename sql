-- Assuming you have two tables 'x' and 'y' with the same column structure

DECLARE
   batch_size NUMBER := 1000; -- Set the desired batch size
   total_rows NUMBER;
   batch_number NUMBER := 1;
BEGIN
   -- Get the total number of rows in table 'x'
   SELECT COUNT(*) INTO total_rows FROM x;
   
   WHILE batch_size * (batch_number - 1) < total_rows LOOP
      -- Insert data into table 'y' in batches with all columns
      INSERT INTO y
      SELECT *
      FROM x
      WHERE rownum <= batch_size
      OFFSET batch_size * (batch_number - 1);
      
      -- Commit after each batch to free resources and persist data
      COMMIT;

      -- Increment the batch number
      batch_number := batch_number + 1;
   END LOOP;
   
   -- Final commit outside the loop
   COMMIT;
END;
/
