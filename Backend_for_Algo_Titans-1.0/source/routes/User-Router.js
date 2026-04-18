const express = require('express')
const router = express.Router()
const { CreateChat } = require('../controllers/Chat-Controller')
const IsLoggedIn = require('../middlewares/IsLoggedIn')
const { CreateAccount, Login } = require('../controllers/Auth-Controller')

router.get("/",(req,res) => {
 res.redirect("/create_account") 
})

router.get("/create_account",(req,res) => {
 res.render("create_account") 
})

router.post("/create_account", CreateAccount)

router.get("/login",(req,res) => {
 res.render("login") 
})

router.post("/login", Login)

router.get("/user_exists",(req,res) => {
 res.render("user_exists") 
})

router.get("/something_went_wrong",(req,res) => {
 res.render("something_went_wrong") 
})

router.post('/chat', IsLoggedIn, CreateChat)


module.exports = router 