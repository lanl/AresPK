.. _dev:

====================
Ares Developer Guide
====================

Logging
========

We offer a few helper functions for logging within the Ares codebase, and rely on Plog as our logger.
Plog is initialized with the log level specified by the command line argument "--log-level <level>" with the possibilities
verbose, debug, info, warning, error, and fatal, in that order of severity. The default is info, and everything logged at or 
above the severity level chosen in the command line will be displayed, while everything lower will be ignored.  
To use Plog, "#include <plog/Log.h>" in your file and then run 

.. code-block::

    PLOG(plog::<severity_level>) << "Put your logging here";  

Logging tasks within the driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To log tasks in the driver function, the best practice is below:

.. code-block::

    auto log_task =
        (utils::ShouldLog(partition) ? task_list.AddTask(dependency, [partition]() {
            PLOG(plog::debug) << TaskInfo(partition) << "Logging here";
            return TaskStatus::complete;
        }) : none);
          
The ShouldLog function ensures we only add this logging task once, but can take in true for "log_per_process" and 
"log_per_partition" as well. The TaskInfo function adds the calling process id and partition to the log. 

.. doxygenindex::
