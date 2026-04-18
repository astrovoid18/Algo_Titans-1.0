require('dotenv').config()


const express = require('express')
const cookieparser = require('cookie-parser')
const path = require('path') 

const app = express()  
app.set("view engine", "ejs")
const port = 3000 


const db = require('./config/Mongoose-Connection')


const UserRouter = require('./routes/User-Router')

app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(cookieparser())
app.use(express.static(path.join(__dirname, "public")))

app.use("/", UserRouter)

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
 