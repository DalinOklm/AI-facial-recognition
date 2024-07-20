const express = require('express');
const multer = require('multer');
const fs = require('fs');
const fsPromises = fs.promises; // Promises-based fs functions
const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData, loadImage } = canvas;
require('dotenv').config();
const bodyParser = require('body-parser');
const twilio = require('twilio');
const http = require('http');
const socketIo = require('socket.io');
const session = require('express-session');
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const flash = require('express-flash');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const qrcode = require('qrcode');
const moment = require('moment');
const dotenv = require('dotenv');
const geolib = require('geolib');
let result = dotenv.config();
const csv = require('csv-writer').createObjectCsvWriter;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

app.use(express.urlencoded({ extended: true }));
// Create a new Express app
app.use(bodyParser.json());

const dbHost = process.env.DB_HOST;
const dbUser = process.env.DB_USERNAME;
const dbPassword = process.env.DB_PASSWORD;
const dbName = process.env.DB_DATABASE;

app.set('view engine', 'ejs');
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
  res.render('index');  
	//res.render('rsvp');
	//res.render('login');
});


// Set up body-parser middleware
app.use(bodyParser.urlencoded({ extended: true }));

// Set up MySQL connection pool
const pool = mysql.createPool({
  host: dbHost,
  user: dbUser,
  password: dbPassword,
  database: dbName,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// Set up session middleware
app.use(session({
  secret: 'secret',
  resave: false,
  saveUninitialized: false
}));
app.use(passport.initialize());
app.use(passport.session());
app.use(flash());

// Middleware to check for a session
app.use((req, res, next) => {
  if (req.session.user || req.path === '/logins' || req.path === '/register') {
    next();
  } else {
    res.redirect('/logins');
  }
});

// Middleware to check for a session and pass user data to the views
app.use((req, res, next) => {
  res.locals.user = req.session.user; // Pass the user data to the views
  next();
});


// ============================= NEW CODE ====================================
// =================================================================
// =================================================================

                  const accountSid = process.env.TWILIO_ACCOUNT_SID;
                  const authToken = process.env.TWILIO_AUTH_TOKEN;
                  const client = twilio(accountSid, authToken);

                  app.post('/send-sms', (req, res) => {
                    const { to, message } = req.body;

                    // client.messages
                    //   .create({
                    //     body: message,
                    //     from: process.env.TWILIO_PHONE_NUMBER,
                    //     to: to
                    //   })
                    //   .then(message => {
                    //     console.log(`SMS sent: ${message.sid}`);
                    //     res.send('SMS sent successfully');
                    //   })
                    //   .catch(error => {
                    //     console.error('Error sending SMS:', error);
                    //     res.status(500).send('Error sending SMS');
                    //   });
                  });



                  // Load face-api.js models when server starts
                  async function loadFaceApiModels() {
                    console.log('Loading face-api.js models...');
                    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'public', 'models'));
                    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'public', 'models'));
                    await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'public', 'models'));
                    console.log('Face-api.js models loaded.');
                  }

                  loadFaceApiModels();

                const storage = multer.diskStorage({
                destination: (req, file, cb) => {
                  const label = req.query.label;
                  const dir = path.join(__dirname, 'public', 'uploads', label);
                  console.log(`Checking existence of directory: ${dir}`);
                  if (!fs.existsSync(dir)) {
                    console.log(`Directory does not exist. Creating: ${dir}`);
                    fs.mkdirSync(dir, { recursive: true });
                  }
                  cb(null, dir);
                },
                filename: (req, file, cb) => {
                  const filename = `${Date.now()}.jpg`;
                  console.log(`Saving file as: ${filename}`);
                  cb(null, filename);
                }
                });

                const upload = multer({ storage });

                app.post('/upload', upload.single('image'), (req, res) => {
                console.log('Upload route hit. Files uploaded:');
                console.log('file: ', req.file);
                res.send('Image uploaded');
                });

                let labeledFaceDescriptors = [];

                async function loadLabeledImages() {
                try {
                  const labels = await fsPromises.readdir(path.join(__dirname, 'public', 'uploads'));
                  console.log('Labels found:', labels);

                  for (const label of labels) {
                    const descriptors = [];
                    const images = await fsPromises.readdir(path.join(__dirname, 'public', 'uploads', label));
                    console.log('Images found for label', label, ':', images);

                    for (const image of images) {
                      const imgPath = path.join(__dirname, 'public', 'uploads', label, image);
                      console.log(`Loading image: ${imgPath}`);
                      const img = await loadImageFromFile(imgPath);
                      console.log("done with loadImageFromFile");

                      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                      console.log("done with detections");
                      if (detections) {
                        descriptors.push(detections.descriptor);
                      }
                    }

                    if (descriptors.length > 0) {
                      labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
                    }
                  }
                  console.log('Processed labeled face descriptors:', labeledFaceDescriptors);
                  return labeledFaceDescriptors;
                } catch (error) {
                  console.error('Error loading labeled images:', error);
                }
                }

                async function loadImageFromFile(filePath) {
                try {
                  console.log(`Reading image file: ${filePath}`);
                  const img = await loadImage(filePath);
                  console.log(`Image loaded: ${filePath}`);
                  return img;
                } catch (error) {
                  console.error(`Error reading image file: ${filePath}`, error);
                  throw error;
                }
                }

              // Serve HTML files
              // app.get('/', (req, res) => {
              // res.sendFile(path.join(__dirname, 'public', 'index.html'));
              // });

            app.get('/register', (req, res) => {
            res.sendFile(path.join(__dirname, 'public', 'register.html'));
            });

            app.get('/real-time-face-recognition', async (req, res) => {
                    // Ensure labeled face descriptors are loaded before serving the page
                    if (labeledFaceDescriptors.length === 0) {
                      console.log('No labeled face descriptors found. Loading from images...');
                      labeledFaceDescriptors = await loadLabeledImages();
                      console.log('Labeled face descriptors loaded.');
                    } else {
                      console.log('Labeled face descriptors already loaded.');
                    }
                    res.sendFile(path.join(__dirname, 'public', 'real_time_face_recognition.html'));
            });

            app.get('/get-labeled-faces', (req, res) => {
            console.log('Sending labeled face descriptors to client.');
            res.json(labeledFaceDescriptors.map(desc => ({
              label: desc.label,
              descriptors: desc.descriptors.map(d => Array.from(d))
            })));
            });



// =================================================================
// =================================================================
// ================================== NEW CODE ===============================


io.on('connection', (socket) => {
  console.log('A user connected');

          // app.get('/checkin-success', (req, res) => {
          //   var message = "new user 2023";
          //   io.emit('chatMessage', message); // Broadcast message to all clients
          //   res.render('checkin-success.ejs');
          // });

          //--------------------------------------

           app.get('/checkin-success', (req, res) => {
            // Get the user id from the query parameter
            const userId = req.query.user;
          
            // Query the database to get the user_name and check_in based on the id
            pool.query(
              'SELECT user_name, check_in FROM mfp_users ' +
              'JOIN mfp_check_ins ON mfp_users.id = mfp_check_ins.user_id ' +
              'WHERE mfp_users.id = ?'+
              'LIMIT 5',
              [userId],
              (err, result) => {
                if (err) {
                  console.error('Database query error:', err);
                  res.redirect('/checkin-success'); // Handle error as needed
                  return;
                }
          
                // Get the user name and check_in datetime from the result
                const userName = result[0].user_name;
                const user_id = result[0].id;
                const checkInDateTimeUTC = new Date(result[0].check_in);
    
                // Convert the UTC date to local date
                const checkInDateTime = checkInDateTimeUTC.toLocaleString();
                
                console.log('result: ', result);
                console.log('checkInDateTime: ', checkInDateTime);

                // Split the datetime into date and time parts
                const [datePart, timePart] = checkInDateTime.split(' ');
                
                console.log('User:', userName);
                console.log('userId:', userId);
                console.log('Date:', datePart);
                console.log('Time:', timePart);
          
                // Customize the message with the retrieved user name and formatted datetime
                const message = `New user: ${userName} has checked in on ${datePart} at ${timePart}`;
          
                console.log('Customized message:', message);
          
                // Emit a newCheckIn event to all connected clients with separate fields
                io.emit('chatMessage', {
                  userName,
                  userId,
                  checkInDate: datePart,
                  checkInTime: timePart
                });
          
                // Render the view with the user name
                //res.render('checkin-success.ejs');
                 // Render the view and pass the results to it
                res.render('checkin-success.ejs', { check_ins_history: result});
              }
            );
          });
          
          
          

          //--------------------------------------

          app.get('/checkout-success', (req, res) => {
            // Get the user id from the query parameter
            const userId = req.query.user;

            // Query the database to get the user_name and check_in based on the id
            pool.query(
              'SELECT user_name, check_out FROM mfp_users ' +
              'JOIN mfp_check_outs ON mfp_users.id = mfp_check_outs.user_id ' +
              'WHERE mfp_users.id = ?',
              [userId],
              (err, result) => {
                if (err) {
                  console.error('Database query error:', err);
                  res.redirect('/checkout-success'); // Handle error as needed
                  return;
                }
          
                // Get the user name and check_in datetime from the result
                const userName = result[0].user_name;
                const checkOutDateTimeUTC = new Date(result[0].check_out);
    
                // Convert the UTC date to local date
                const checkOutDateTime = checkOutDateTimeUTC.toLocaleString();
                
                console.log('result: ', result);
                console.log('checkOutDateTime: ', checkOutDateTime);

                // Split the datetime into date and time parts
                const [datePart, timePart] = checkOutDateTime.split(' ');
                
                console.log('User:', userName);
                console.log('Date:', datePart);
                console.log('Time:', timePart);
          
                // Customize the message with the retrieved user name and formatted datetime
                const message = `New user: ${userName} has checked out on ${datePart} at ${timePart}`;
          
                console.log('Customized message:', message);
          
                // Emit a newCheckIn event to all connected clients with separate fields
                io.emit('check_out_notification', {
                  userName,
                  checkInDate: datePart,
                  checkInTime: timePart
                });
          
                // Render the view with the user name
                res.render('checkout-success.ejs');
              }
            );
          });

          // Render the registration form
          app.get('/register', (req, res) => {
              res.render('register.ejs');
          });


          // Process the registration form
          app.post('/register', (req, res) => {
            const { user_name, email,password } = req.body;

            // Hash the password
            bcrypt.hash(password, 10, (err, hash) => {
                if (err) {
                    console.log(err);
                    res.redirect('/register');
                    return;
                }

                // Insert the user into the database
                pool.query('INSERT INTO mfp_users (user_name, email, password) VALUES (?, ?, ?)', [user_name, email,hash], (err, result) => {
                    if (err) {
                        console.log(err);
                        res.redirect('/register');
                        return;
                    }

                    res.redirect('/logins');
                });
            });
          });

          //login route
          //app.get('/logins', (req, res) => {
            //res.render('login');
          //});

          //login function
          app.post('/logins', (req, res) => {
            const { user_name, password } = req.body;

            pool.query('SELECT * FROM mfp_users WHERE user_name = ?', [user_name], async (error, results) => {
              if (error) {
                console.log(error);
                res.redirect('/logins');
                return;
              }

              if (results.length === 0) {
                req.flash('error', 'Incorrect username');
                res.redirect('/logins');
                return;
              }

              const user = results[0];
              const match = await bcrypt.compare(password, user.password);

              if (!match) {
                req.flash('error', 'Incorrect password');
                res.redirect('/logins');
                return;
              }

              req.session.user = user;
              res.redirect('/dashboard');
            });
          });

          // Process the check-out form
          app.post('/check_out', (req, res) => {
            // Insert the check-out record into the database

            const { user_id, latitude, longitude } = req.body;

            // user coordinate
            const userLatitude = latitude;
            const userLongitude = longitude;

             // MFP location accurate
             const targetLatitude = -26.06181;
             const targetLongitude = 28.08786;
             //const radius = 0.05; // 50 meters radius
             const radius = 20000; // 10km radius

             function isWithinRadius(latitude, longitude, targetLatitude, targetLongitude, radius) {
              const distance = geolib.getDistance(
                { latitude, longitude },
                { latitude: targetLatitude, longitude: targetLongitude }
              );

              return distance <= radius;
            }


             const withinRadius = isWithinRadius(userLatitude, userLongitude, targetLatitude, targetLongitude, radius);
             //const withinRadius = true;

            const currentDate = new Date();
            const johannesburgOffset = 0; // Johannesburg is UTC+2
            const johannesburgTime = new Date(currentDate.getTime() + johannesburgOffset * 60 * 60 * 1000);
            const year = johannesburgTime.getFullYear();
            const month = String(johannesburgTime.getMonth() + 1).padStart(2, '0');
            const day = String(johannesburgTime.getDate()).padStart(2, '0');
            const hours = String(johannesburgTime.getHours()).padStart(2, '0');
            const minutes = String(johannesburgTime.getMinutes()).padStart(2, '0');
            const seconds = String(johannesburgTime.getSeconds()).padStart(2, '0');

            const check_out = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;

            if (withinRadius) {
                    pool.query(
                      'INSERT INTO mfp_check_outs (user_id, check_out) VALUES (?, ?)',
                      [user_id, check_out],
                      (err, result) => {
                        if (err) {
                          console.log(err);
                          res.redirect('/check');
                          return;
                        }

                        //res.redirect('/checkout-success');
                        res.redirect(`/checkout-success?user=${encodeURIComponent(user_id)}`);
                      }
                    );
          }else{
            const currentDate = moment().format('YYYY-MM-DD');
            const userId = req.session.user.id; // Assuming user_id is available in the session
            const {latitude, longitude } = req.body;

            // user coordinate
            const userLatitude = latitude;
            const userLongitude = longitude;

            // Display a warning message to the user
            console.log("wrong location");
            //res.render('dashboard.ejs', { checkOuts: results=[], message: '1', userLatitudefront: userLatitude, userLongitudefront: userLongitude});

            pool.query(
              'SELECT * FROM mfp_check_ins WHERE DATE(check_in) = ? AND user_id = ?',
              [currentDate, userId],
              (err, results) => {
                if (err) {
                  console.log(err);
                  return;
                }

                // Fetch the list of check-in users for the current day
                pool.query(
                  `SELECT a.id, a.check_in, b.user_name 
                   FROM mfp_check_ins a 
                   JOIN mfp_users b ON a.user_id = b.id 
                   LEFT JOIN mfp_check_outs c ON a.user_id = c.user_id AND DATE(a.check_in) = DATE(c.check_out)
                   WHERE DATE(a.check_in) = ? AND c.id IS NULL`,
                  [currentDate],
                  (checkInErr, checkInResults) => {
                    if (checkInErr) {
                      console.log(checkInErr);
                      return;
                    }

                      // Emit a newCheckIn event to all connected clients with separate fields
                      // io.emit('checkInUsers', {
                      //   checkInResults
                      // });

                      console.log('\n');
                      //console.log('Check Outs:', checkOuts); 
                      console.log('\ncurrentDate:', currentDate);
                      console.log('\nuserId:', userId);
                      console.log('\nCheck In Users:', checkInResults);
                      console.log('\nUser Latitude:', userLatitude);
                      console.log('\nUser Longitude:', userLongitude);
                      console.log('\n');

                      // Render the EJS template and pass the checkOuts array
                res.render('dashboard', { checkOuts: results=['1'], checkInUsers: checkInResults, message: '1',  userLatitudefront: userLatitude, userLongitudefront: userLongitude });


                  }
                );
              }
            );

          }
          });
        

          // Render the check_in.ejs form
          app.get('/check_in', (req, res) => {
            res.render('check_in.ejs');
          });



          app.get('/api/check-in-status', (req, res) => {
            const { user_id } = req.query;
            const currentDate = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format
        
            pool.query(
                'SELECT COUNT(*) as count FROM mfp_check_ins WHERE user_id = ? AND DATE(check_in) = ?',
                [user_id, currentDate],
                (err, results) => {
                    if (err) {
                        console.error(err);
                        return res.status(500).json({ error: 'Internal Server Error' });
                    }
        
                    const alreadyCheckedIn = results[0].count > 0;
                    res.json({ checkedIn: alreadyCheckedIn });
                }
            );
        });



          //   socket.on('disconnect', () => {
          //     console.log('User disconnected');
          //   });
            
          // });

          // Process the check-in form
          app.post('/check_in', (req, res) => {
            // Insert the check-in record into the database
            const { user_id, user_name, latitude, longitude } = req.body;
        
            console.log("user_id: ", user_id);
            console.log("latitude: ", latitude);
            console.log("longitude: ", longitude);
            console.log("user_name: ", user_name);
        
            // user coordinate
            const userLatitude = latitude;
            const userLongitude = longitude;
        
            // MFP location accurate
            const targetLatitude = -26.06181;
            const targetLongitude = 28.08786;
            const radius = 2000; // 2km radius
        
            function isWithinRadius(latitude, longitude, targetLatitude, targetLongitude, radius) {
                const distance = geolib.getDistance(
                    { latitude, longitude },
                    { latitude: targetLatitude, longitude: targetLongitude }
                );
                return distance <= radius;
            }
        
            // const withinRadius = isWithinRadius(userLatitude, userLongitude, targetLatitude, targetLongitude, radius);
            const withinRadius = true;
        
            const currentDate = new Date();
            const johannesburgOffset = 0; // Johannesburg is UTC+2
            const johannesburgTime = new Date(currentDate.getTime() + johannesburgOffset * 60 * 60 * 1000);
            const year = johannesburgTime.getFullYear();
            const month = String(johannesburgTime.getMonth() + 1).padStart(2, '0');
            const day = String(johannesburgTime.getDate()).padStart(2, '0');
            const hours = String(johannesburgTime.getHours()).padStart(2, '0');
            const minutes = String(johannesburgTime.getMinutes()).padStart(2, '0');
            const seconds = String(johannesburgTime.getSeconds()).padStart(2, '0');
        
            const check_in = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        
            console.log("user_id: ", user_id, "\n");
            console.log("latitude", latitude, "\n");
            console.log("longitude: ", longitude, "\n");
            console.log("withinRadius: ", withinRadius);
            console.log("check_in: ", check_in);
        
            if (withinRadius) {
                let userIdValue = user_id;
                let userNameValue = user_name;
        
                if (!user_id && user_name) {
                    pool.query(
                        'SELECT id FROM mfp_users WHERE user_name = ?',
                        [user_name],
                        (err, results) => {
                            if (err) {
                                console.log(err);
                                res.redirect('/dashboard');
                                return;
                            }
                            if (results.length > 0) {
                                userIdValue = results[0].id;
                            } else {
                                console.log("User not found");
                                res.redirect('/dashboard');
                                return;
                            }
        
                            // Insert the check-in record into the database using user_id
                            insertCheckIn(userIdValue, check_in, res);
                        }
                    );
                } else {
                    // Insert the check-in record into the database using user_id
                    insertCheckIn(userIdValue, check_in, res);
                }
            } else {
                // Display a warning message to the user
                console.log("wrong location");
                res.render('dashboard.ejs', { checkOuts: [], message: '1', userLatitudefront: userLatitude, userLongitudefront: userLongitude });
            }
        });
        
        function insertCheckIn(userId, check_in, res) {
            pool.query(
                'INSERT INTO mfp_check_ins (user_id, check_in) VALUES (?, ?)',
                [userId, check_in],
                (err, result) => {
                    if (err) {
                        console.log(err);
                        res.redirect('/dashboard');
                        return;
                    }
        
                    // Emit a newCheckIn event to all connected clients
                    const newCheckIn = {
                        user_name: userId, // Replace with the actual user's name
                        check_in: new Date().toISOString() // Current timestamp
                    };
        
                    // Redirect to checkin-success with the user name as a query parameter
                    res.redirect(`/checkin-success?user=${encodeURIComponent(userId)}`);
                }
            );
        }
        


                // Route to render the view with check-in records and user names
              app.get('/check_ins', (req, res) => {
                //const currentDate = moment().format('YYYY-MM-DD'); // Get the current date in 'YYYY-MM-DD' format
                const currentDate = "2023-09-15";

                const checkInQuery = 'SELECT mfp_check_ins.id, mfp_check_ins.check_in, mfp_users.user_name FROM mfp_check_ins JOIN mfp_users ON mfp_check_ins.user_id = mfp_users.id ORDER BY mfp_check_ins.id DESC';

                const newQuery = `SELECT u.user_name, 
                      IFNULL(ci.check_in, 'Absent') as check_in, 
                      co.check_out,
                      IFNULL(TIMESTAMPDIFF(HOUR, ci.check_in, co.check_out), '') as total_hours
                    FROM mfp_users u
                    LEFT JOIN mfp_check_ins ci ON u.id = ci.user_id AND DATE(ci.check_in) = ?
                    LEFT JOIN mfp_check_outs co ON u.id = co.user_id AND DATE(co.check_out) = ?
                    ORDER BY ci.id DESC;`;

                // New query to count the number of check-ins for the current day
                const checkInCountQuery = `SELECT COUNT(*) as check_in_count FROM mfp_check_ins WHERE DATE(check_in) = ?`;

                // Execute the first SQL query to retrieve check-ins
                pool.query(checkInQuery, (err, checkIns) => {
                  if (err) {
                    console.log(err);
                    res.redirect('/dashboard');
                    return;
                  }

                  // Execute the second SQL query
                  pool.query(newQuery, [currentDate, currentDate], (err, newQueryResults) => {
                    if (err) {
                      console.log(err);
                      res.redirect('/dashboard');
                      return;
                    }

                    // Execute the third SQL query to get the check-in count
                    pool.query(checkInCountQuery, [currentDate], (err, checkInCountResults) => {
                      if (err) {
                        console.log(err);
                        res.redirect('/dashboard');
                        return;
                      }

                      // Combine the results of both queries into a single object
                      const dataToSend_ = {
                        check_ins: checkIns,
                        new_query_results: newQueryResults,
                        check_in_count: checkInCountResults[0].check_in_count, // Access the count from the result
                      };

                      // Render the 'check_ins' view and pass the combined data to it
                      res.render('check_ins', { dataToSend: dataToSend_ });

                      // Log what is sent to the client-side
                      console.log(dataToSend_);
                    });
                  });
                });
              });




              app.post('/search', (req, res) => {
               // Extract data from the request body
              const { searchValue, startDate, endDate } = req.body;

              // Debugging: Log the received data
              console.log('Received searchValue:', searchValue);
              console.log('Received startDate:', startDate);
              console.log('Received endDate:', endDate);

              // Function to format a date in 'YYYY-MM-DD' format
              const formatDate = (dateString) => {
                const [month, day, year] = dateString.split('/');
                return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
              };

              // Format the start and end dates if they exist
              const formattedStartDate = startDate ? formatDate(startDate) : null;
              const formattedEndDate = endDate ? formatDate(endDate) : null;

              // Construct the SQL query
              const query = `
                SELECT u.user_name, ci.user_id, ci.id as "check-in id", ci.check_in, co.id as "check-out id", co.check_out,
                TIMESTAMPDIFF(HOUR, ci.check_in, co.check_out) as total_hours
                FROM mfp_users u
                LEFT JOIN mfp_check_ins ci ON u.id = ci.user_id
                LEFT JOIN mfp_check_outs co ON u.id = co.user_id AND DATE(ci.check_in) = DATE(co.check_out)
                WHERE ci.check_in BETWEEN ${formattedStartDate ? `'${formattedStartDate}'` : 'null'} AND ${formattedEndDate ? `'${formattedEndDate}'` : 'null'}
                AND u.user_name LIKE '${searchValue}%'
                ORDER BY u.user_name ASC;
              `;

              // Debugging: Log the constructed SQL query
              console.log('Constructed SQL Query:', query);

              // Execute the SQL query
              pool.query(query, (err, results) => {
                if (err) {
                  console.error(err);
                  return res.status(500).json({ error: 'An error occurred while fetching data.' });
                }

                console.log('Data:', results);

                // Send the query results as a JSON response
                res.json({ data: results });
              });

              });
              
              



          // Route to render the view with check-in records and user names
          app.get('/check_outs', (req, res) => {
            const sql = 'SELECT mfp_check_outs.id, mfp_check_outs.check_out, mfp_users.user_name FROM mfp_check_outs JOIN mfp_users ON mfp_check_outs.user_id = mfp_users.id ORDER BY mfp_check_outs.check_out DESC';

            // Execute the SQL query
            pool.query(sql, (err, results) => {
              if (err) {
                console.log(err);
                res.redirect('/dashboard');
                return;
              }

              // Render the view and pass the results to it
              res.render('check_outs', { check_outs: results });
            });
          });

          // app.get('/dashboard', (req, res) => {

          //     // Query the database to get check-outs created today
          //   const currentDate = moment().format('YYYY-MM-DD');
          //   const userId = req.session.user.id; // Assuming user_id is available in the session

          //   pool.query(
          //     'SELECT * FROM mfp_check_ins WHERE DATE(check_in) = ? AND user_id = ?',
          //     [currentDate, userId],
          //     (err, results) => {
          //       if (err) {
          //         console.log(err);
          //         return;
          //       }

          //       // Render the EJS template and pass the checkOuts array
          //       res.render('dashboard', { checkOuts: results, message: '',  userLatitudefront: '', userLongitudefront: ''  });
          //     }
          //   );
          // });


          app.get('/dashboard', async (req, res) => {
            try {
              const currentDate = moment().format('YYYY-MM-DD');
              const userId = req.session.user.id; // Assuming user_id is available in the session
          
              const checkIns = await new Promise((resolve, reject) => {
                pool.query(
                  'SELECT * FROM mfp_check_ins WHERE DATE(check_in) = ? AND user_id = ?',
                  [currentDate, userId],
                  (err, results) => {
                    if (err) return reject(err);
                    resolve(results);
                  }
                );
              });
          
              const checkInUsers = await new Promise((resolve, reject) => {
                pool.query(
                  `SELECT a.id, a.check_in, b.user_name 
                   FROM mfp_check_ins a 
                   JOIN mfp_users b ON a.user_id = b.id 
                   LEFT JOIN mfp_check_outs c ON a.user_id = c.user_id AND DATE(a.check_in) = DATE(c.check_out)
                   WHERE DATE(a.check_in) = ? AND c.id IS NULL`,
                  [currentDate],
                  (err, results) => {
                    if (err) return reject(err);
                    resolve(results);
                  }
                );
              });
          
              io.emit('checkInUsers', { checkInUsers });
          
              const customQueryResults = await new Promise((resolve, reject) => {
                pool.query(
                  `SELECT u.user_name, ci.check_in, co.check_out, 
                    TIMESTAMPDIFF(HOUR, ci.check_in, co.check_out) as total_hours 
                    FROM mfp_users u 
                    LEFT JOIN mfp_check_ins ci ON u.id = ci.user_id 
                    LEFT JOIN mfp_check_outs co ON u.id = co.user_id AND DATE(ci.check_in) = DATE(co.check_out) 
                    WHERE ci.user_id = ? 
                    ORDER BY DATE(ci.check_in) DESC 
                    LIMIT 5;`,
                  [userId],
                  (err, results) => {
                    if (err) return reject(err);
                    resolve(results);
                  }
                );
              });
          
              if (labeledFaceDescriptors.length === 0) {
                console.log('No labeled face descriptors found. Loading from images...');
                labeledFaceDescriptors = await loadLabeledImages();
                console.log('Labeled face descriptors loaded.');
              } else {
                console.log('Labeled face descriptors already loaded.');
              }
          
              res.render('dashboard', {
                currentDate_client: currentDate,
                checkOuts: checkIns,
                checkInUsers: checkInUsers,
                customQueryResults: customQueryResults, // Pass the custom query results to the template
                message: '',
                userLatitudefront: '',
                userLongitudefront: ''
              });
            } catch (err) {
              console.error(err);
              res.status(500).send('Internal Server Error');
            }
          });
          



          // Logout route
          app.get('/logout', (req, res) => {
            // Destroy the session and logout the user
            req.session.destroy(err => {
              if (err) {
                console.log(err);
                res.redirect('/');
                return;
              }

              // Redirect the user to the desired page after successful logout
              res.redirect('/logins');
            });
          });

          app.get('/generate_checkin_csv', (req, res) => {

            const startDate = req.query.start;
            const endDate = req.query.end;
          
            console.log(`Received request to generate CSV for date range: ${startDate} to ${endDate}`);
          
            const sql = `SELECT u.email, u.user_name, ci.check_in, co.check_out
                         FROM mfp_check_ins ci
                         JOIN mfp_users u ON ci.user_id = u.id
                         LEFT JOIN mfp_check_outs co ON ci.user_id = co.user_id AND DATE(ci.check_in) = DATE(co.check_out)
                         WHERE DATE(ci.check_in) BETWEEN '${startDate}' AND '${endDate}'
                         ORDER BY ci.id ASC`;
          
            pool.query(sql, (err, results) => {
              if (err) {
                console.log(err);
                res.redirect('/dashboard');
                return;
              }
          
              const data = results.map(check_in => {
                const checkInTime = new Date(check_in.check_in);
                const checkOutTime = check_in.check_out ? new Date(check_in.check_out) : null;
                const timeDifferenceHours = checkOutTime ? ((checkOutTime - checkInTime) / 3600000).toFixed(2) : 'N/A';
          
                return {
                  Email: check_in.email || 'Email not found',
                  UserName: check_in.user_name || 'User Name not found',
                  CheckInDate: checkInTime.toLocaleDateString(),
                  CheckInTime: checkInTime.toLocaleTimeString(),
                  CheckOutTime: checkOutTime ? checkOutTime.toLocaleTimeString() : 'Not Checked Out',
                  TimeDifference: timeDifferenceHours
                };
              });
          
              const csvFilePath = `./${startDate.replace(/-/g, '_')}_to_${endDate.replace(/-/g, '_')}_checkin.csv`;
          
              const csvWriter = csv({
                path: csvFilePath,
                header: [
                  { id: 'Email', title: 'Email' },
                  { id: 'UserName', title: 'User Name' },
                  { id: 'CheckInDate', title: 'Check In Date' },
                  { id: 'CheckInTime', title: 'Check In Time' },
                  { id: 'CheckOutTime', title: 'Check Out Time' },
                  { id: 'TimeDifference', title: 'Time Difference (hours)' }
                ]
              });
          
              csvWriter.writeRecords(data)
                .then(() => {
                  console.log('CSV file created successfully');
                  res.download(csvFilePath, `${startDate}_to_${endDate}_check_ins.csv`, () => {
                    fs.unlinkSync(csvFilePath);
                  });
                })
                .catch(error => {
                  console.error('Error creating CSV file:', error);
                  res.redirect('/dashboard');
                });
            });
          });
          


                    // Route to generate a CSV file of check-in records
                    app.get('/generate_checkout_csv', (req, res) => {

                      //const currentDate = new Date().toLocaleDateString(); // Get current date in the format 'MM/DD/YYYY'
            
                      const currentDate = new Date().toISOString().split('T')[0]; // Get current date in the format 'YYYY-MM-DD'
            
                      //const sql = 'SELECT mfp_check_ins.id, mfp_check_ins.check_in, mfp_users.user_name FROM mfp_check_ins JOIN mfp_users ON mfp_check_ins.user_id = mfp_users.id ORDER BY mfp_check_ins.id DESC';
            
                      const sql = `SELECT mfp_check_outs.id, mfp_check_outs.check_out, mfp_users.user_name FROM mfp_check_outs JOIN mfp_users ON mfp_check_outs.user_id = mfp_users.id WHERE DATE(mfp_check_outs.check_out) = '${currentDate}' ORDER BY mfp_check_outs.id DESC`;
            
                      // Execute the SQL query
                      pool.query(sql, (err, results) => {
                        if (err) {
                          console.log(err);
                          res.redirect('/dashboard');
                          return;
                        }
            
                        // Extract the required fields from the results
                        const data = results.map(check_in => {
                          return {
                            Name: check_in.user_name || 'user_name not found',
                            Date: check_in.check_in.toLocaleDateString(),
                            Time: check_in.check_in.toLocaleTimeString()
                          };
                        });
            
                        // Set the CSV file path and name
                        const csvFilePath = `./${currentDate.replace(/\//g, '-')}_checkout.csv`;
            
                        // Create the CSV writer and define the header
                        const csvWriter = csv({
                          path: csvFilePath,
                          header: [
                            { id: 'Name', title: 'Name' },
                            { id: 'Date', title: 'Date' },
                            { id: 'Time', title: 'Time' }
                          ]
                        });
            
                        // Write the data to the CSV file
                        csvWriter.writeRecords(data)
                          .then(() => {
                            console.log('CSV file created successfully');
                            res.download(csvFilePath, currentDate+'_checkout.csv', () => {
                              // Remove the CSV file after download
                              fs.unlinkSync(csvFilePath);
                            });
                          })
                          .catch(error => {
                            console.error('Error creating CSV file:', error);
                            res.redirect('/dashboard');
                          });
                      });
                    });


          socket.on('disconnect', () => {
            console.log('User disconnected');
          });

});

 //login route
   app.get('/logins', (req, res) => {
       res.render('login');
     });




const port = process.env.PORT || 8000;
server.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});